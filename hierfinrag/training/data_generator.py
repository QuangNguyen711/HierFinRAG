"""
Training Data Generator for TTGNN using LLM.

This module generates diverse query-node pairs for supervised contrastive learning.

Data generation strategy:
1. STRUCTURED PATTERNS: Create different query types systematically
   - Specific value lookups (cell queries)
   - Descriptive questions (paragraph queries)
   - Comparison queries (multi-cell queries)
   - Summary/explanation queries (paragraph+cell)
   - Temporal/trend queries (time-series cells)

2. DIVERSITY: Ensure variety in:
   - Node types (cells vs paragraphs)
   - Query complexity (single vs multi-node)
   - Question patterns (what, how much, compare, explain)
   - Content domains (financial metrics, narrative, etc.)

3. QUALITY: Use LLM to generate natural Vietnamese questions
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

from ..parsing.base import Document, Table


@dataclass
class TrainingSample:
    """
    Training sample for supervised contrastive learning.
    
    Structure:
    {
      "id": "sample_001",
      "query": "Lợi nhuận sau thuế năm 2023 là bao nhiêu?",
      "document_id": "doc_vnpt_2023",
      "positive_nodes": ["p2", "t1_r2_c2"],  # 1-3 nodes that answer the question
      "negative_nodes": ["p1", "p3", ...],   # All other leaf nodes
      "positive_content": {
        "p2": "Lợi nhuận sau thuế đạt 3.800 tỷ...",
        "t1_r2_c2": "3.800 tỷ"
      }
    }
    """
    id: str
    query: str
    document_id: str
    positive_nodes: List[str]
    negative_nodes: List[str]
    positive_content: Dict[str, str]


class QueryPattern:
    """Query pattern templates for diverse data generation."""
    
    # Query types and their characteristics
    PATTERNS = {
        'specific_value': {
            'description': 'Query about specific values requiring multiple cells',
            'node_types': ['Cell'],
            'num_nodes': (3, 5),  # Need multiple cells to calculate/compare
            'weight': 0.25
        },
        'descriptive': {
            'description': 'Query about content requiring multiple paragraphs',
            'node_types': ['Paragraph'],
            'num_nodes': (3, 5),  # Need multiple paragraphs for comprehensive answer
            'weight': 0.20
        },
        'comparison': {
            'description': 'Compare multiple values across many cells',
            'node_types': ['Cell'],
            'num_nodes': (4, 7),  # Compare across multiple metrics/years
            'weight': 0.20
        },
        'mixed': {
            'description': 'Combine multiple paragraphs and cells',
            'node_types': ['Paragraph', 'Cell'],
            'num_nodes': (4, 6),  # Mix of 2-3 paragraphs + 2-3 cells
            'weight': 0.20
        },
        'summary': {
            'description': 'Synthesize multiple paragraphs',
            'node_types': ['Paragraph'],
            'num_nodes': (3, 5),  # Need multiple paragraphs to summarize comprehensively
            'weight': 0.15
        }
    }
    
    @staticmethod
    def get_pattern_distribution(num_samples: int) -> List[str]:
        """Generate a distribution of query patterns for diversity."""
        patterns = []
        for pattern_name, config in QueryPattern.PATTERNS.items():
            count = int(num_samples * config['weight'])
            patterns.extend([pattern_name] * count)
        
        # Fill remaining samples
        while len(patterns) < num_samples:
            patterns.append(random.choice(list(QueryPattern.PATTERNS.keys())))
        
        # Shuffle for randomness
        random.shuffle(patterns)
        return patterns[:num_samples]


class TrainingDataGenerator:
    """Generate training data using LLM."""
    
    def __init__(self, env_path: str = ".env"):
        """
        Initialize with LLM credentials.
        
        Args:
            env_path: Path to .env file with LLM credentials
        """
        load_dotenv(env_path)
        
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in .env file")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"✓ LLM Client initialized: {self.model_name}")
        
        # Track used node combinations to avoid duplicates
        self.used_combinations: Set[frozenset] = set()
    
    def test_connection(self) -> bool:
        """Test if LLM API is working with a simple request."""
        print("Testing LLM connection...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Say 'OK'"}
                ],
                max_tokens=5,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            print(f"✓ LLM connection successful! Response: {result}")
            return True
        except Exception as e:
            print(f"✗ LLM connection failed: {e}")
            return False
    
    def extract_leaf_nodes(self, doc: Document) -> Dict[str, Dict[str, Any]]:
        """
        Extract all leaf nodes (paragraphs and cells) from document with rich context.
        
        Returns:
            Dict mapping node_id to {type, text, metadata}
        """
        leaf_nodes = {}
        
        # Extract paragraphs
        for p in doc.paragraphs:
            leaf_nodes[p.id] = {
                'type': 'Paragraph',
                'text': p.text,
                'section_id': p.section_id
            }
        
        # Extract cells with rich context
        for table in doc.tables:
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                
                # Get row/col headers for context
                row_header = table.row_headers[cell.row_idx] if cell.row_idx < len(table.row_headers) else ""
                col_header = table.col_headers[cell.col_idx] if cell.col_idx < len(table.col_headers) else ""
                
                context = f"{table.caption} | {row_header} | {col_header}" if row_header or col_header else table.caption
                
                leaf_nodes[cell_id] = {
                    'type': 'Cell',
                    'text': str(cell.value),
                    'context': context,
                    'table_id': table.id,
                    'row_idx': cell.row_idx,
                    'col_idx': cell.col_idx,
                    'row_header': row_header,
                    'col_header': col_header
                }
        
        return leaf_nodes
    
    def organize_nodes_by_type(self, leaf_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Organize nodes by type for structured sampling."""
        organized = defaultdict(list)
        for node_id, node_data in leaf_nodes.items():
            organized[node_data['type']].append(node_id)
        return organized
    
    def sample_nodes_by_pattern(
        self,
        pattern: str,
        leaf_nodes: Dict[str, Dict[str, Any]],
        nodes_by_type: Dict[str, List[str]],
        max_attempts: int = 50
    ) -> List[str]:
        """
        Sample nodes according to the specified pattern for diversity.
        
        Args:
            pattern: Query pattern type
            leaf_nodes: All leaf nodes
            nodes_by_type: Nodes organized by type
            max_attempts: Max attempts to find unique combination
            
        Returns:
            List of selected node IDs
        """
        config = QueryPattern.PATTERNS[pattern]
        min_nodes, max_nodes = config['num_nodes']
        required_types = config['node_types']
        
        for _ in range(max_attempts):
            selected = []
            
            if pattern == 'specific_value':
                # Multiple cells for calculation/comparison
                cells = nodes_by_type.get('Cell', [])
                if len(cells) < min_nodes:
                    continue
                num = random.randint(min_nodes, min(max_nodes, len(cells)))
                selected = random.sample(cells, num)
            
            elif pattern == 'descriptive':
                # Multiple paragraphs for comprehensive answer
                paragraphs = nodes_by_type.get('Paragraph', [])
                if len(paragraphs) < min_nodes:
                    continue
                num = random.randint(min_nodes, min(max_nodes, len(paragraphs)))
                selected = random.sample(paragraphs, num)
            
            elif pattern == 'comparison':
                # Many cells for multi-way comparison
                cells = nodes_by_type.get('Cell', [])
                if len(cells) < min_nodes:
                    continue
                
                num = random.randint(min_nodes, min(max_nodes, len(cells)))
                
                # Try to get cells from same or related tables for meaningful comparison
                tables_with_cells = defaultdict(list)
                for cell_id in cells:
                    table_id = leaf_nodes[cell_id].get('table_id')
                    if table_id:
                        tables_with_cells[table_id].append(cell_id)
                
                # Pick cells from 1-2 tables if possible
                valid_tables = [t for t, c in tables_with_cells.items() if len(c) >= min_nodes]
                if valid_tables:
                    if len(valid_tables) > 1 and random.random() < 0.3:
                        # Mix from 2 tables
                        table1, table2 = random.sample(valid_tables, 2)
                        n1 = num // 2
                        n2 = num - n1
                        selected = (random.sample(tables_with_cells[table1], min(n1, len(tables_with_cells[table1]))) +
                                   random.sample(tables_with_cells[table2], min(n2, len(tables_with_cells[table2]))))
                    else:
                        # From one table
                        table = random.choice(valid_tables)
                        selected = random.sample(tables_with_cells[table], min(num, len(tables_with_cells[table])))
                else:
                    # Fallback: any cells
                    selected = random.sample(cells, num)
            
            elif pattern == 'mixed':
                # Multiple paragraphs + multiple cells
                paragraphs = nodes_by_type.get('Paragraph', [])
                cells = nodes_by_type.get('Cell', [])
                
                if not paragraphs or not cells:
                    continue
                
                # Aim for 2-3 paragraphs + 2-3 cells
                num = random.randint(min_nodes, max_nodes)
                num_paras = random.randint(2, min(3, len(paragraphs), num - 2))
                num_cells = num - num_paras
                
                if num_cells > len(cells) or num_paras > len(paragraphs):
                    continue
                    
                selected = random.sample(paragraphs, num_paras) + random.sample(cells, num_cells)
            
            elif pattern == 'summary':
                # Multiple related paragraphs for comprehensive summary
                paragraphs = nodes_by_type.get('Paragraph', [])
                if len(paragraphs) < min_nodes:
                    continue
                
                num = random.randint(min_nodes, min(max_nodes, len(paragraphs)))
                
                # Try to get paragraphs from same section for coherence
                sections = defaultdict(list)
                for p_id in paragraphs:
                    section = leaf_nodes[p_id].get('section_id')
                    sections[section].append(p_id)
                
                # Pick section with enough paragraphs if possible
                valid_sections = [s for s, ps in sections.items() if len(ps) >= min_nodes]
                if valid_sections and random.random() < 0.7:
                    section = random.choice(valid_sections)
                    selected = random.sample(sections[section], min(num, len(sections[section])))
                else:
                    # Mix from multiple sections
                    selected = random.sample(paragraphs, num)
            
            # Check uniqueness
            combination = frozenset(selected)
            if combination not in self.used_combinations:
                self.used_combinations.add(combination)
                return selected
        
        # If max attempts reached, just return last attempt
        return selected
    
    def generate_question_from_nodes(
        self,
        positive_nodes: Dict[str, Dict[str, Any]],
        pattern: str,
        doc_title: str,
        temperature: float = 0.8
    ) -> str:
        """
        Use LLM to generate a natural question based on pattern and content.
        
        Args:
            positive_nodes: Dict of selected positive node data
            pattern: Query pattern type
            doc_title: Document title for context
            temperature: LLM temperature
            
        Returns:
            Generated question
        """
        # Build pattern-specific instruction
        pattern_instructions = {
            'specific_value': "Tạo câu hỏi cần tính toán/tổng hợp NHIỀU giá trị số. VD: 'Tổng X và Y là bao nhiêu?', 'Tính tỷ lệ giữa A, B và C?', 'So sánh các chỉ số X, Y, Z?'",
            'descriptive': "Tạo câu hỏi cần TỔNG HỢP nhiều đoạn văn. VD: 'Phân tích toàn bộ hoạt động X?', 'Mô tả các khía cạnh của Y?', 'Công ty thực hiện những gì về Z?'",
            'comparison': "Tạo câu hỏi so sánh NHIỀU giá trị/chỉ số. VD: 'So sánh tất cả các chỉ số X, Y, Z qua các năm?', 'Phân tích xu hướng của A, B, C, D?'",
            'mixed': "Tạo câu hỏi cần KẾT HỢP nhiều thông tin văn bản và số liệu. VD: 'Phân tích nguyên nhân X dựa trên các chỉ số?', 'Giải thích Y kết hợp dữ liệu và bối cảnh?'",
            'summary': "Tạo câu hỏi tổng hợp/phân tích NHIỀU khía cạnh. VD: 'Tổng quan toàn diện về X?', 'Phân tích đa chiều về Y?', 'Đánh giá tổng thể Z?'"
        }
        
        instruction = pattern_instructions.get(pattern, "Tạo câu hỏi tự nhiên về nội dung")
        
        # Build content context
        content_parts = []
        for node_id, node_data in positive_nodes.items():
            if node_data['type'] == 'Paragraph':
                content_parts.append(f"Đoạn văn: {node_data['text'][:200]}")
            else:  # Cell
                context = node_data.get('context', '')
                row_header = node_data.get('row_header', '')
                col_header = node_data.get('col_header', '')
                value = node_data['text']
                
                if row_header and col_header:
                    content_parts.append(f"Ô bảng [{row_header}][{col_header}]: {value}")
                elif context:
                    content_parts.append(f"Ô bảng ({context}): {value}")
                else:
                    content_parts.append(f"Ô bảng: {value}")
        
        content = "\n".join(content_parts)
        
        prompt = f"""Bạn là chuyên gia tài chính. Từ báo cáo "{doc_title}", hãy tạo câu hỏi dựa trên nội dung sau.

NỘI DUNG:
{content}

HƯỚNG DẪN:
{instruction}

⚠️ YÊU CẦU QUAN TRỌNG:
- Câu hỏi PHẢI cần TẤT CẢ các thông tin trên để trả lời đầy đủ
- Không thể trả lời chỉ bằng 1 thông tin, phải kết hợp nhiều thông tin
- Câu hỏi tự nhiên, phức tạp, như người dùng thực sự sẽ hỏi
- Ngắn gọn nhưng đòi hỏi phân tích/tổng hợp/so sánh nhiều dữ liệu
- Chỉ trả về câu hỏi, không giải thích
- Tiếng Việt

CÂU HỎI:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia tài chính tạo câu hỏi cho hệ thống training."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=200,
                max_completion_tokens=200
            )
            
            question = response.choices[0].message.content.strip()
            question = question.strip('"').strip("'")
            return question
            
        except Exception as e:
            print(f"    ⚠ LLM error: {e}")
            return self._generate_fallback_question(positive_nodes, pattern, doc_title)
    
    def _generate_fallback_question(
        self, 
        positive_nodes: Dict[str, Dict[str, Any]],
        pattern: str,
        doc_title: str
    ) -> str:
        """Fallback question generation without LLM."""
        node_types = [n['type'] for n in positive_nodes.values()]
        first_node = list(positive_nodes.values())[0]
        
        if pattern == 'specific_value' and 'Cell' in node_types:
            templates = [
                "Giá trị của {} là bao nhiêu?",
                "Số liệu {} trong báo cáo là gì?",
                "{} đạt mức nào?"
            ]
            key = first_node.get('row_header', first_node.get('context', 'chỉ tiêu'))
        
        elif pattern == 'comparison' and 'Cell' in node_types:
            templates = [
                "So sánh {} giữa các năm?",
                "Sự thay đổi của {} như thế nào?",
                "Tăng trưởng {} là bao nhiêu?"
            ]
            key = first_node.get('row_header', 'chỉ tiêu')
        
        elif pattern == 'descriptive' and 'Paragraph' in node_types:
            templates = [
                "Mô tả về {}?",
                "Thông tin chi tiết về {}?",
                "Công ty báo cáo gì về {}?"
            ]
            key = first_node['text'][:30]
        
        else:
            templates = [
                "Thông tin về {} trong báo cáo?",
                "Dữ liệu về {} như thế nào?",
                "Kết quả {} là gì?"
            ]
            key = first_node['text'][:30] if 'text' in first_node else 'chỉ tiêu tài chính'
        
        template = random.choice(templates)
        return template.format(key)
    
    def generate_sample(
        self, 
        doc: Document,
        sample_id: int,
        pattern: str,
        all_leaf_nodes: Dict[str, Dict[str, Any]],
        nodes_by_type: Dict[str, List[str]]
    ) -> TrainingSample:
        """
        Generate a single training sample with specified pattern.
        
        Args:
            doc: Parsed document
            sample_id: Sample ID
            pattern: Query pattern type
            all_leaf_nodes: All leaf nodes in document
            nodes_by_type: Nodes organized by type
            
        Returns:
            TrainingSample instance
        """
        # Sample nodes according to pattern
        positive_node_ids = self.sample_nodes_by_pattern(
            pattern, 
            all_leaf_nodes, 
            nodes_by_type
        )
        
        if not positive_node_ids:
            raise ValueError(f"Could not sample nodes for pattern: {pattern}")
        
        # All other nodes are negative
        negative_node_ids = [
            nid for nid in all_leaf_nodes.keys() 
            if nid not in positive_node_ids
        ]
        
        # Get positive node content
        positive_nodes = {nid: all_leaf_nodes[nid] for nid in positive_node_ids}
        positive_content = {nid: all_leaf_nodes[nid]['text'] for nid in positive_node_ids}
        
        # Generate question using LLM
        query = self.generate_question_from_nodes(
            positive_nodes, 
            pattern,
            doc.title
        )
        
        return TrainingSample(
            id=f"sample_{sample_id:04d}",
            query=query,
            document_id=doc.id,
            positive_nodes=positive_node_ids,
            negative_nodes=negative_node_ids,
            positive_content=positive_content
        )
    
    def generate_dataset(
        self,
        doc: Document,
        num_samples: int = 1000,
        output_path: str = "data/training_data.json"
    ) -> List[TrainingSample]:
        """
        Generate complete diverse training dataset.
        
        Args:
            doc: Parsed document
            num_samples: Number of samples to generate
            output_path: Where to save the dataset
            
        Returns:
            List of training samples
        """
        print(f"\n{'='*80}")
        print(f"Generating {num_samples} DIVERSE training samples...")
        print(f"{'='*80}\n")
        
        # Extract and organize nodes once
        all_leaf_nodes = self.extract_leaf_nodes(doc)
        nodes_by_type = self.organize_nodes_by_type(all_leaf_nodes)
        
        print(f"Document statistics:")
        print(f"  - Total leaf nodes: {len(all_leaf_nodes)}")
        print(f"  - Paragraphs: {len(nodes_by_type.get('Paragraph', []))}")
        print(f"  - Cells: {len(nodes_by_type.get('Cell', []))}")
        
        # Get diverse pattern distribution
        patterns = QueryPattern.get_pattern_distribution(num_samples)
        pattern_counts = defaultdict(int)
        for p in patterns:
            pattern_counts[p] += 1
        
        print(f"\nQuery pattern distribution:")
        for pattern, count in sorted(pattern_counts.items()):
            pct = (count / num_samples) * 100
            print(f"  - {pattern}: {count} ({pct:.1f}%)")
        print()
        
        samples = []
        failed = 0
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            try:
                pattern = patterns[i]
                sample = self.generate_sample(
                    doc=doc,
                    sample_id=i,
                    pattern=pattern,
                    all_leaf_nodes=all_leaf_nodes,
                    nodes_by_type=nodes_by_type
                )
                samples.append(sample)
                
                # Show first few samples with pattern info
                if i < 5:
                    print(f"\n[Sample {i+1}] Pattern: {pattern}")
                    print(f"  Query: {sample.query}")
                    print(f"  Positive: {len(sample.positive_nodes)} nodes - {sample.positive_nodes[:3]}")
                    print(f"  Negative: {len(sample.negative_nodes)} nodes")
                
            except Exception as e:
                if i < 10:
                    print(f"\n⚠ Error generating sample {i}: {e}")
                failed += 1
                continue
        
        # Save to file if output_path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [asdict(s) for s in samples], 
                    f, 
                    ensure_ascii=False, 
                    indent=2
                )
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {len(samples)} samples ({failed} failed)")
        print(f"✓ Unique combinations: {len(self.used_combinations)}")
        if output_path:
            print(f"✓ Saved to: {output_path}")
        print(f"{'='*80}\n")
        
        # Print pattern distribution of actual samples
        actual_pattern_counts = defaultdict(int)
        for s in samples:
            # Infer pattern from sample
            num_pos = len(s.positive_nodes)
            if num_pos == 1:
                types = [all_leaf_nodes[nid]['type'] for nid in s.positive_nodes]
                if 'Cell' in types:
                    actual_pattern_counts['specific_value'] += 1
                else:
                    actual_pattern_counts['descriptive'] += 1
            elif num_pos >= 2:
                types = set([all_leaf_nodes[nid]['type'] for nid in s.positive_nodes])
                if len(types) > 1:
                    actual_pattern_counts['mixed'] += 1
                elif 'Cell' in types:
                    actual_pattern_counts['comparison'] += 1
                else:
                    actual_pattern_counts['summary'] += 1
        
        print("Actual pattern distribution in generated samples:")
        for pattern, count in sorted(actual_pattern_counts.items()):
            pct = (count / len(samples)) * 100 if samples else 0
            print(f"  - {pattern}: {count} ({pct:.1f}%)")
        print()
        
        return samples


def load_training_data(file_path: str) -> List[TrainingSample]:
    """
    Load training data from JSON file.
    
    Args:
        file_path: Path to training data JSON file
        
    Returns:
        List of TrainingSample objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        sample = TrainingSample(
            id=item['id'],
            query=item['query'],
            document_id=item['document_id'],
            positive_nodes=item['positive_nodes'],
            negative_nodes=item['negative_nodes'],
            positive_content=item['positive_content']
        )
        samples.append(sample)
    
    return samples
