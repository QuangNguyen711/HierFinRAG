"""
Synthetic Vietnamese Financial Document Generator using LLM.

Generates diverse financial reports for training data augmentation.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os


class SyntheticDocumentGenerator:
    """Generate diverse synthetic Vietnamese financial documents using LLM."""
    
    # Vietnamese company templates
    COMPANIES = [
        {"name": "VNPT", "full": "Tập đoàn Bưu chính Viễn thông Việt Nam", "sector": "Viễn thông"},
        {"name": "Viettel", "full": "Tập đoàn Công nghiệp - Viễn thông Quân đội", "sector": "Viễn thông"},
        {"name": "FPT", "full": "Tập đoàn FPT", "sector": "Công nghệ thông tin"},
        {"name": "MobiFone", "full": "Tổng Công ty Viễn thông MobiFone", "sector": "Viễn thông"},
        {"name": "VietinBank", "full": "Ngân hàng TMCP Công Thương Việt Nam", "sector": "Ngân hàng"},
        {"name": "Vietcombank", "full": "Ngân hàng TMCP Ngoại thương Việt Nam", "sector": "Ngân hàng"},
        {"name": "Vinamilk", "full": "Công ty Cổ phần Sữa Việt Nam", "sector": "Thực phẩm"},
        {"name": "Vingroup", "full": "Tập đoàn Vingroup", "sector": "Đa ngành"},
        {"name": "Hòa Phát", "full": "Tập đoàn Hòa Phát", "sector": "Thép"},
        {"name": "PetroVietnam", "full": "Tập đoàn Dầu khí Quốc gia Việt Nam", "sector": "Năng lượng"}
    ]
    
    YEARS = [2020, 2021, 2022, 2023, 2024]
    
    # Diverse paragraph topics for each section type
    PARAGRAPH_TOPICS = {
        "Tổng Quan": [
            "Doanh thu và tăng trưởng thị trường",
            "Sản phẩm/dịch vụ mới ra mắt",
            "Mở rộng thị trường và khách hàng",
            "Đối thủ cạnh tranh và vị thế thị trường",
            "Chuyển đổi số và công nghệ mới",
            "Chiến lược kinh doanh tổng thể"
        ],
        "Kết Quả": [
            "Chi tiết doanh thu theo mảng",
            "Phân tích lợi nhuận và biên lợi nhuận",
            "Chi phí hoạt động và tối ưu hóa",
            "So sánh với năm trước và kế hoạch",
            "Yếu tố tác động đến kết quả",
            "Triển vọng ngắn hạn"
        ],
        "Tài Sản": [
            "Cơ cấu tài sản và thanh khoản",
            "Đầu tư dài hạn và tài sản cố định",
            "Nguồn vốn chủ sở hữu",
            "Nợ vay và quản lý nợ",
            "Vốn lưu động và chu kỳ kinh doanh",
            "Tỷ lệ nợ và khả năng thanh toán"
        ],
        "Phân Tích": [
            "Các chỉ số sinh lời (ROA, ROE, ROS)",
            "Hiệu quả sử dụng tài sản",
            "Cấu trúc vốn và đòn bẩy tài chính",
            "Khả năng thanh toán và thanh khoản",
            "So sánh với trung bình ngành",
            "Xu hướng các chỉ số qua các năm"
        ],
        "Định Hướng": [
            "Mục tiêu doanh thu và lợi nhuận",
            "Kế hoạch đầu tư và mở rộng",
            "Chiến lược sản phẩm/dịch vụ",
            "Thị trường mục tiêu mới",
            "Đối tác chiến lược và M&A",
            "Nghiên cứu và phát triển"
        ],
        "Rủi Ro": [
            "Rủi ro thị trường và cạnh tranh",
            "Rủi ro tài chính và tỷ giá",
            "Rủi ro vận hành và công nghệ",
            "Rủi ro chính sách và pháp lý",
            "Biện pháp quản trị rủi ro",
            "Bảo hiểm và dự phòng"
        ],
        "Phát Triển Bền Vững": [
            "Trách nhiệm môi trường",
            "Chính sách nhân sự và đào tạo",
            "Đóng góp xã hội và cộng đồng",
            "Quản trị công ty minh bạch",
            "An toàn lao động và sức khỏe",
            "Mục tiêu phát triển bền vững"
        ]
    }
    
    def __init__(self, env_path: str = ".env"):
        """Initialize LLM client."""
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
        
        print(f"✓ Synthetic Document Generator initialized with {self.model_name}")
    
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
    
    def generate_document(
        self,
        doc_id: str,
        company: Dict[str, str] = None,
        year: int = None,
        num_sections: int = None,
        num_tables: int = None
    ) -> Dict[str, Any]:
        """
        Generate a single synthetic financial document.
        
        Args:
            doc_id: Document identifier
            company: Company info dict, random if None
            year: Fiscal year, random if None
            num_sections: Number of sections (3-7), random if None
            num_tables: Number of tables (2-5), random if None
            
        Returns:
            Document dict in the same format as mock_vietnamese_financial.json
        """
        # Random selections
        if company is None:
            company = random.choice(self.COMPANIES)
        if year is None:
            year = random.choice(self.YEARS)
        if num_sections is None:
            num_sections = random.randint(3, 7)
        if num_tables is None:
            num_tables = random.randint(2, 5)
        
        print(f"  Generating: {company['name']} {year} ({num_sections} sections, {num_tables} tables)")
        
        # Generate ONE consistent set of metrics for the entire document
        doc_metrics = self._generate_random_metrics(company['sector'], year)
        
        # Generate document structure
        doc = {
            "id": doc_id,
            "title": f"Báo Cáo Tài Chính {company['name']} {year}",
            "sections": [],
            "paragraphs": [],
            "tables": []
        }
        
        # Generate sections and content
        section_titles = self._generate_section_titles(company, year, num_sections)
        
        for i, section_title in enumerate(section_titles):
            section_id = f"s{i+1}"
            
            # Generate paragraphs for this section (2-4 per section)
            num_paragraphs = random.randint(2, 4)
            paragraph_ids = [f"p{len(doc['paragraphs']) + j + 1}" for j in range(num_paragraphs)]
            
            # Maybe add a table to this section (probability based on num_tables)
            table_id = None
            if len(doc['tables']) < num_tables and random.random() < 0.7:
                table_id = f"t{len(doc['tables']) + 1}"
            
            # Build content_ids
            content_ids = []
            
            # Interleave paragraphs and table
            for j, p_id in enumerate(paragraph_ids):
                content_ids.append(p_id)
                # Add table after first or second paragraph
                if table_id and j == min(1, len(paragraph_ids) - 1):
                    content_ids.append(table_id)
            
            # Create section
            doc['sections'].append({
                "id": section_id,
                "title": section_title,
                "level": 1,
                "content_ids": content_ids
            })
            
            # Generate paragraphs with topic tracking for diversity
            used_topics = []
            for p_id in paragraph_ids:
                paragraph_text = self._generate_paragraph(
                    company, year, section_title, doc['tables'], used_topics, doc_metrics
                )
                doc['paragraphs'].append({
                    "id": p_id,
                    "text": paragraph_text,
                    "section_id": section_id
                })
            
            # Generate table if needed
            if table_id:
                table = self._generate_table(
                    table_id, company, year, section_title
                )
                doc['tables'].append(table)
        
        return doc
    
    def _generate_section_titles(
        self, 
        company: Dict[str, str], 
        year: int, 
        num_sections: int
    ) -> List[str]:
        """Generate section titles for the document."""
        prompt = f"""Tạo {num_sections} tiêu đề phần (section titles) cho báo cáo tài chính của {company['full']} năm {year}.

YÊU CẦU:
- Tiêu đề phải ngắn gọn, chuyên nghiệp
- Phù hợp với lĩnh vực {company['sector']}
- Đa dạng: tổng quan, kết quả kinh doanh, tài sản, phân tích, định hướng
- Trả về dạng JSON list
- Chỉ trả về JSON, không giải thích

VÍ DỤ:
["Tổng Quan Hoạt Động Kinh Doanh", "Kết Quả Hoạt Động Tài Chính", "Tình Hình Tài Sản và Nguồn Vốn"]

TIÊU ĐỀ ({num_sections} phần):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia tài chính, trả về JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=300,
                max_completion_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            titles = json.loads(content)
            return titles[:num_sections]
            
        except Exception as e:
            print(f"    ⚠ LLM error generating titles: {e}, using fallback")
            return self._fallback_section_titles(num_sections)
    
    def _fallback_section_titles(self, num_sections: int) -> List[str]:
        """Fallback section titles if LLM fails."""
        all_titles = [
            "Tổng Quan Hoạt Động Kinh Doanh",
            "Kết Quả Hoạt Động Tài Chính",
            "Tình Hình Tài Sản và Nguồn Vốn",
            "Phân Tích Chỉ Số Tài Chính",
            "Định Hướng Phát Triển",
            "Quản Trị Rủi Ro",
            "Phát Triển Bền Vững"
        ]
        return random.sample(all_titles, min(num_sections, len(all_titles)))
    
    def _get_topic_category(self, section_title: str) -> str:
        """Map section title to topic category."""
        title_lower = section_title.lower()
        
        if any(kw in title_lower for kw in ["tổng quan", "hoạt động", "kinh doanh"]):
            return "Tổng Quan"
        elif any(kw in title_lower for kw in ["kết quả", "doanh thu", "lợi nhuận"]):
            return "Kết Quả"
        elif any(kw in title_lower for kw in ["tài sản", "nguồn vốn", "cân đối"]):
            return "Tài Sản"
        elif any(kw in title_lower for kw in ["phân tích", "chỉ số", "hiệu quả"]):
            return "Phân Tích"
        elif any(kw in title_lower for kw in ["định hướng", "chiến lược", "phát triển năm"]):
            return "Định Hướng"
        elif any(kw in title_lower for kw in ["rủi ro", "an toàn", "bảo mật"]):
            return "Rủi Ro"
        elif any(kw in title_lower for kw in ["bền vững", "môi trường", "trách nhiệm"]):
            return "Phát Triển Bền Vững"
        else:
            # Default to Tổng Quan if no match
            return "Tổng Quan"
    
    def _generate_paragraph(
        self,
        company: Dict[str, str],
        year: int,
        section_title: str,
        existing_tables: List[Dict],
        used_topics: List[str] = None,
        doc_metrics: Dict[str, Any] = None
    ) -> str:
        """Generate a paragraph of content with specific topic focus."""
        # Get topic category and select a specific topic
        topic_category = self._get_topic_category(section_title)
        available_topics = self.PARAGRAPH_TOPICS.get(topic_category, self.PARAGRAPH_TOPICS["Tổng Quan"])
        
        # Avoid repeating topics in same section
        if used_topics is None:
            used_topics = []
        
        remaining_topics = [t for t in available_topics if t not in used_topics]
        if not remaining_topics:
            remaining_topics = available_topics  # Cycle through if exhausted
        
        specific_topic = random.choice(remaining_topics)
        used_topics.append(specific_topic)
        
        # DECISION: Should this paragraph include financial metrics?
        # Some topics should be purely qualitative
        qualitative_keywords = [
            "sản phẩm", "dịch vụ", "thị trường", "đối thủ", "công nghệ", 
            "chuyển đổi số", "rủi ro", "đối tác", "môi trường", "nhân sự",
            "nghiên cứu", "phát triển", "quản trị"
        ]
        
        include_metrics = True
        for keyword in qualitative_keywords:
            if keyword in specific_topic.lower():
                # 60% chance to be qualitative only
                if random.random() < 0.6:
                    include_metrics = False
                break
        
        # Company-specific context
        sector_context = {
            "Viễn thông": "dịch vụ di động, internet, và chuyển đổi số",
            "Ngân hàng": "cho vay, huy động vốn, và dịch vụ tài chính",
            "Công nghệ thông tin": "phần mềm, dịch vụ IT, và giải pháp công nghệ",
            "Năng lượng": "khai thác, chế biến, và phân phối năng lượng",
            "Đa ngành": "bất động sản, bán lẻ, và các lĩnh vực liên quan",
            "Thép": "sản xuất và kinh doanh thép xây dựng",
            "Thực phẩm": "sản xuất và kinh doanh thực phẩm tiêu dùng"
        }.get(company['sector'], "các lĩnh vực kinh doanh")
        
        # Build metrics section for prompt (optional)
        metrics_text = ""
        if include_metrics and doc_metrics:
            metrics_text = f"\n\nMỘT SỐ SỐ LIỆU (nếu phù hợp với chủ đề):\n{json.dumps(doc_metrics, ensure_ascii=False, indent=2)}"
        
        # Varied prompt structures
        prompt_templates = [
            # Template 1: Topic-focused
            f"""Viết 1 đoạn văn (2-3 câu) về "{specific_topic}" trong bối cảnh {company['name']} năm {year}.

CONTEXT:
- Công ty: {company['full']} - lĩnh vực {company['sector']}
- Chuyên về: {sector_context}
- Chủ đề: {specific_topic}
{metrics_text}

YÊU CẦU:
- TẬP TRUNG VÀO {specific_topic}, KHÔNG chung chung
- {'Mô tả định tính, ít số liệu' if not include_metrics else 'Kết hợp số liệu nếu liên quan'}
- Cụ thể về {company['sector']}, đề cập sản phẩm/dịch vụ thực tế
- ĐA DẠNG cấu trúc câu, không bắt đầu với "Năm {year}, {company['name']}..."
- 50-120 từ, phong cách chuyên nghiệp
- Chỉ trả về đoạn văn

ĐOẠN VĂN:""",
            
            # Template 2: Challenge-focus
            f"""Viết đoạn văn phân tích "{specific_topic}" của {company['name']} năm {year}, nhấn mạnh thách thức và cơ hội.

CONTEXT:
- {company['full']} - {company['sector']} ({sector_context})
- Focus: {specific_topic}
{metrics_text}

YÊU CẦU:
- Nội dung SÂU về {specific_topic}, không lặp lại
- {'Phân tích định tính' if not include_metrics else 'Số liệu minh họa nếu cần'}
- Đề cập thách thức/cơ hội/giải pháp cụ thể
- Cấu trúc câu ĐA DẠNG
- Chỉ trả về đoạn văn

ĐOẠN VĂN:""",
            
            # Template 3: Action-oriented
            f"""Mô tả hành động/sáng kiến của {company['name']} về "{specific_topic}" trong năm {year}.

- Lĩnh vực: {company['sector']} - {sector_context}
- Góc nhìn: {specific_topic}
{metrics_text}

YÊU CẦU:
- Tập trung hành động/chiến lược CỤ THỂ
- {'Mô tả chất lượng' if not include_metrics else 'Kết quả có thể đo lường'}  
- TRÁNH câu sáo rỗng, chung chung
- Đa dạng cách diễn đạt

ĐOẠN VĂN:"""
        ]
        
        prompt = random.choice(prompt_templates)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia viết báo cáo tài chính với khả năng tạo nội dung đa dạng, cụ thể và sâu sắc."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.95,  # Higher for more diversity
                max_tokens=250,
                max_completion_tokens=250
            )
            
            paragraph = response.choices[0].message.content.strip()
            return paragraph
            
        except Exception as e:
            print(f"    ⚠ LLM error generating paragraph: {e}")
            return self._fallback_paragraph(company, year, section_title, doc_metrics or {}, specific_topic)
    
    def _generate_random_metrics(self, sector: str, year: int) -> Dict[str, Any]:
        """Generate random but realistic financial metrics."""
        prev_year = year - 1
        
        # Base ranges depend on sector
        if sector in ["Viễn thông", "Ngân hàng", "Công nghệ thông tin"]:
            revenue_base = random.randint(30000, 80000)
        elif sector in ["Năng lượng", "Đa ngành", "Thép"]:
            revenue_base = random.randint(50000, 150000)
        else:
            revenue_base = random.randint(20000, 60000)
        
        growth_rate = random.uniform(5, 20)
        prev_revenue = int(revenue_base / (1 + growth_rate/100))
        
        profit_margin = random.uniform(5, 15)
        profit = int(revenue_base * profit_margin / 100)
        prev_profit = int(prev_revenue * profit_margin / 100)
        
        return {
            "doanh_thu": f"{revenue_base:,} tỷ đồng",
            "doanh_thu_nam_truoc": f"{prev_revenue:,} tỷ đồng",
            "tang_truong": f"{growth_rate:.1f}%",
            "loi_nhuan": f"{profit:,} tỷ đồng",
            "loi_nhuan_nam_truoc": f"{prev_profit:,} tỷ đồng",
            "bien_loi_nhuan": f"{profit_margin:.2f}%"
        }
    
    def _fallback_paragraph(
        self,
        company: Dict[str, str],
        year: int,
        section_title: str,
        metrics: Dict,
        specific_topic: str = None
    ) -> str:
        """Fallback paragraph generation with diverse templates."""
        # Topic-specific templates for diversity
        if specific_topic and "sản phẩm" in specific_topic.lower():
            templates = [
                f"Năm {year}, {company['name']} ra mắt nhiều sản phẩm/dịch vụ mới, góp phần tăng trưởng doanh thu {metrics['tang_truong']}. Các sản phẩm này đáp ứng nhu cầu thị trường và củng cố vị thế cạnh tranh.",
                f"{company['name']} đầu tư mạnh vào nghiên cứu và phát triển sản phẩm trong năm {year}. Kết quả là danh mục sản phẩm đa dạng hơn, đóng góp vào tăng trưởng {metrics['tang_truong']}."
            ]
        elif specific_topic and "thị trường" in specific_topic.lower():
            templates = [
                f"Chiến lược mở rộng thị trường của {company['name']} đạt kết quả tích cực trong năm {year}. Công ty tăng cường hiện diện tại các khu vực mới, góp phần tăng trưởng doanh thu {metrics['tang_truong']}.",
                f"{company['name']} nâng cao thị phần trong lĩnh vực {company['sector']} nhờ chiến lược tiếp cận khách hàng hiệu quả. Năm {year}, công ty đạt doanh thu {metrics['doanh_thu']}, tăng {metrics['tang_truong']}."
            ]
        elif specific_topic and any(kw in specific_topic.lower() for kw in ["chi phí", "tối ưu"]):
            templates = [
                f"Các biện pháp tối ưu chi phí của {company['name']} trong năm {year} mang lại hiệu quả rõ rệt. Biên lợi nhuận cải thiện lên {metrics['bien_loi_nhuan']}, phản ánh quản trị hiệu quả.",
                f"{company['name']} kiểm soát chi phí hoạt động chặt chẽ, duy trì biên lợi nhuận ở mức {metrics['bien_loi_nhuan']}. Điều này giúp công ty duy trì lợi thế cạnh tranh trong năm {year}."
            ]
        elif specific_topic and any(kw in specific_topic.lower() for kw in ["đầu tư", "tài sản"]):
            templates = [
                f"Năm {year}, {company['name']} đẩy mạnh đầu tư vào hạ tầng và công nghệ. Việc này tạo nền tảng cho tăng trưởng bền vững và nâng cao hiệu quả hoạt động.",
                f"Cơ cấu tài sản của {company['name']} được tối ưu hóa trong năm {year}. Công ty tập trung vào các khoản đầu tư mang lại giá trị dài hạn và cải thiện năng lực cạnh tranh."
            ]
        elif specific_topic and any(kw in specific_topic.lower() for kw in ["nhân sự", "đào tạo"]):
            templates = [
                f"{company['name']} chú trọng phát triển nguồn nhân lực trong năm {year}. Các chương trình đào tạo và chính sách đãi ngộ hấp dẫn giúp thu hút và giữ chân nhân tài.",
                f"Chính sách nhân sự của {company['name']} trong năm {year} tập trung vào nâng cao năng lực đội ngũ. Điều này đóng góp trực tiếp vào hiệu quả hoạt động và doanh thu tăng {metrics['tang_truong']}."
            ]
        else:
            # Generic templates as fallback
            templates = [
                f"Năm {year}, {company['name']} đạt doanh thu {metrics['doanh_thu']}, tăng {metrics['tang_truong']} so với năm {year-1}. Lợi nhuận sau thuế đạt {metrics['loi_nhuan']}, cho thấy sự tăng trưởng ổn định.",
                f"Kết quả kinh doanh năm {year} của {company['name']} ghi nhận doanh thu {metrics['doanh_thu']}. Biên lợi nhuận đạt {metrics['bien_loi_nhuan']}, phản ánh hiệu quả hoạt động tốt.",
                f"{company['name']} tiếp tục duy trì đà tăng trưởng trong năm {year} với doanh thu {metrics['doanh_thu']}. Công ty đặt mục tiêu mở rộng thị phần và nâng cao chất lượng dịch vụ."
            ]
        return random.choice(templates)
    
    def _generate_table(
        self,
        table_id: str,
        company: Dict[str, str],
        year: int,
        section_title: str
    ) -> Dict[str, Any]:
        """Generate a financial table."""
        # Determine table type
        table_types = [
            "kết quả kinh doanh",
            "cơ cấu doanh thu",
            "bảng cân đối kế toán",
            "chỉ số tài chính"
        ]
        
        # Pick based on section context or random
        if "kết quả" in section_title.lower() or "kinh doanh" in section_title.lower():
            table_type = "kết quả kinh doanh"
        elif "doanh thu" in section_title.lower():
            table_type = "cơ cấu doanh thu"
        elif "tài sản" in section_title.lower() or "nguồn vốn" in section_title.lower():
            table_type = "bảng cân đối kế toán"
        elif "phân tích" in section_title.lower() or "chỉ số" in section_title.lower():
            table_type = "chỉ số tài chính"
        else:
            table_type = random.choice(table_types)
        
        if table_type == "kết quả kinh doanh":
            return self._generate_income_statement_table(table_id, year)
        elif table_type == "cơ cấu doanh thu":
            return self._generate_revenue_breakdown_table(table_id, year, company)
        elif table_type == "bảng cân đối kế toán":
            return self._generate_balance_sheet_table(table_id, year)
        else:
            return self._generate_financial_ratios_table(table_id, year)
    
    def _generate_income_statement_table(self, table_id: str, year: int) -> Dict:
        """Generate income statement table."""
        prev_year = year - 1
        
        # Generate realistic numbers
        revenue_curr = random.randint(30000, 80000)
        revenue_prev = int(revenue_curr / random.uniform(1.05, 1.20))
        
        gross_profit_curr = int(revenue_curr * random.uniform(0.10, 0.15))
        gross_profit_prev = int(revenue_prev * random.uniform(0.10, 0.15))
        
        net_profit_curr = int(gross_profit_curr * random.uniform(0.75, 0.85))
        net_profit_prev = int(gross_profit_prev * random.uniform(0.75, 0.85))
        
        growth_revenue = ((revenue_curr - revenue_prev) / revenue_prev * 100)
        growth_gross = ((gross_profit_curr - gross_profit_prev) / gross_profit_prev * 100)
        growth_net = ((net_profit_curr - net_profit_prev) / net_profit_prev * 100)
        
        return {
            "id": table_id,
            "caption": f"Bảng {table_id.replace('t', '')}: Kết quả kinh doanh tổng hợp {prev_year}-{year}",
            "col_headers": ["Chỉ tiêu", f"Năm {prev_year}", f"Năm {year}", "Tăng trưởng (%)"],
            "row_headers": ["Tổng doanh thu", "Lợi nhuận trước thuế", "Lợi nhuận sau thuế"],
            "cells": [
                {"row": 0, "col": 0, "value": "Tổng doanh thu", "is_header": True},
                {"row": 0, "col": 1, "value": f"{revenue_prev:,} tỷ", "is_header": False},
                {"row": 0, "col": 2, "value": f"{revenue_curr:,} tỷ", "is_header": False},
                {"row": 0, "col": 3, "value": f"{growth_revenue:.1f}%", "is_header": False},
                {"row": 1, "col": 0, "value": "Lợi nhuận trước thuế", "is_header": True},
                {"row": 1, "col": 1, "value": f"{gross_profit_prev:,} tỷ", "is_header": False},
                {"row": 1, "col": 2, "value": f"{gross_profit_curr:,} tỷ", "is_header": False},
                {"row": 1, "col": 3, "value": f"{growth_gross:.1f}%", "is_header": False},
                {"row": 2, "col": 0, "value": "Lợi nhuận sau thuế", "is_header": True},
                {"row": 2, "col": 1, "value": f"{net_profit_prev:,} tỷ", "is_header": False},
                {"row": 2, "col": 2, "value": f"{net_profit_curr:,} tỷ", "is_header": False},
                {"row": 2, "col": 3, "value": f"{growth_net:.1f}%", "is_header": False}
            ]
        }
    
    def _generate_revenue_breakdown_table(self, table_id: str, year: int, company: Dict) -> Dict:
        """Generate revenue breakdown by service/product."""
        # Generate 3-5 revenue streams
        num_streams = random.randint(3, 5)
        
        # Common revenue streams by sector
        streams_by_sector = {
            "Viễn thông": ["Viễn thông di động", "Internet băng rộng", "Giải pháp CNTT", "Dịch vụ khác"],
            "Ngân hàng": ["Tín dụng", "Dịch vụ thanh toán", "Đầu tư chứng khoán", "Thu dịch vụ khác"],
            "Công nghệ thông tin": ["Phần mềm", "Giải pháp doanh nghiệp", "Dịch vụ đám mây", "Phần cứng"],
            "Thực phẩm": ["Sản phẩm A", "Sản phẩm B", "Sản phẩm C", "Sản phẩm khác"],
            "Năng lượng": ["Dầu thô", "Khí thiên nhiên", "Điện", "Dịch vụ khác"],
            "Thép": ["Thép xây dựng", "Thép cuộn", "Thép ống", "Sản phẩm khác"]
        }
        
        sector = company['sector']
        if sector in streams_by_sector:
            available_streams = streams_by_sector[sector]
        else:
            available_streams = ["Sản phẩm chính", "Dịch vụ", "Kinh doanh khác", "Thu nhập khác"]
        
        stream_names = random.sample(available_streams, min(num_streams, len(available_streams)))
        
        # Generate revenue values that sum to ~100%
        weights = [random.uniform(0.5, 3.0) for _ in range(num_streams)]
        total_weight = sum(weights)
        percentages = [(w / total_weight) * 100 for w in weights]
        
        # Generate absolute values
        total_revenue = random.randint(30000, 80000)
        revenues = [int(total_revenue * (p / 100)) for p in percentages]
        
        cells = [
            {"row": 0, "col": 0, "value": "Loại dịch vụ", "is_header": True},
            {"row": 0, "col": 1, "value": "Doanh thu (tỷ đồng)", "is_header": True},
            {"row": 0, "col": 2, "value": "Tỷ trọng (%)", "is_header": True}
        ]
        
        for i, (name, revenue, pct) in enumerate(zip(stream_names, revenues, percentages), start=1):
            cells.extend([
                {"row": i, "col": 0, "value": name, "is_header": False},
                {"row": i, "col": 1, "value": f"{revenue:,}", "is_header": False},
                {"row": i, "col": 2, "value": f"{pct:.1f}%", "is_header": False}
            ])
        
        return {
            "id": table_id,
            "caption": f"Bảng {table_id.replace('t', '')}: Cơ cấu doanh thu năm {year}",
            "col_headers": ["Loại dịch vụ", "Doanh thu (tỷ đồng)", "Tỷ trọng (%)"],
            "row_headers": [],
            "cells": cells
        }
    
    def _generate_balance_sheet_table(self, table_id: str, year: int) -> Dict:
        """Generate balance sheet table."""
        prev_year = year - 1
        
        # Generate realistic balance sheet numbers
        total_assets_curr = random.randint(60000, 150000)
        total_assets_prev = int(total_assets_curr / random.uniform(1.05, 1.15))
        
        current_assets_curr = int(total_assets_curr * random.uniform(0.30, 0.40))
        current_assets_prev = int(total_assets_prev * random.uniform(0.30, 0.40))
        
        fixed_assets_curr = total_assets_curr - current_assets_curr
        fixed_assets_prev = total_assets_prev - current_assets_prev
        
        equity_curr = int(total_assets_curr * random.uniform(0.55, 0.65))
        equity_prev = int(total_assets_prev * random.uniform(0.55, 0.65))
        
        liabilities_curr = total_assets_curr - equity_curr
        liabilities_prev = total_assets_prev - equity_prev
        
        return {
            "id": table_id,
            "caption": f"Bảng {table_id.replace('t', '')}: Bảng cân đối kế toán tóm tắt",
            "col_headers": ["Khoản mục", f"Cuối năm {prev_year}", f"Cuối năm {year}", "Biến động (%)"],
            "row_headers": [],
            "cells": [
                {"row": 0, "col": 0, "value": "Khoản mục", "is_header": True},
                {"row": 0, "col": 1, "value": f"Cuối năm {prev_year}", "is_header": True},
                {"row": 0, "col": 2, "value": f"Cuối năm {year}", "is_header": True},
                {"row": 0, "col": 3, "value": "Biến động (%)", "is_header": True},
                {"row": 1, "col": 0, "value": "Tổng tài sản", "is_header": False},
                {"row": 1, "col": 1, "value": f"{total_assets_prev:,} tỷ", "is_header": False},
                {"row": 1, "col": 2, "value": f"{total_assets_curr:,} tỷ", "is_header": False},
                {"row": 1, "col": 3, "value": f"{((total_assets_curr-total_assets_prev)/total_assets_prev*100):.1f}%", "is_header": False},
                {"row": 2, "col": 0, "value": "Tài sản ngắn hạn", "is_header": False},
                {"row": 2, "col": 1, "value": f"{current_assets_prev:,} tỷ", "is_header": False},
                {"row": 2, "col": 2, "value": f"{current_assets_curr:,} tỷ", "is_header": False},
                {"row": 2, "col": 3, "value": f"{((current_assets_curr-current_assets_prev)/current_assets_prev*100):.1f}%", "is_header": False},
                {"row": 3, "col": 0, "value": "Tài sản dài hạn", "is_header": False},
                {"row": 3, "col": 1, "value": f"{fixed_assets_prev:,} tỷ", "is_header": False},
                {"row": 3, "col": 2, "value": f"{fixed_assets_curr:,} tỷ", "is_header": False},
                {"row": 3, "col": 3, "value": f"{((fixed_assets_curr-fixed_assets_prev)/fixed_assets_prev*100):.1f}%", "is_header": False},
                {"row": 4, "col": 0, "value": "Vốn chủ sở hữu", "is_header": False},
                {"row": 4, "col": 1, "value": f"{equity_prev:,} tỷ", "is_header": False},
                {"row": 4, "col": 2, "value": f"{equity_curr:,} tỷ", "is_header": False},
                {"row": 4, "col": 3, "value": f"{((equity_curr-equity_prev)/equity_prev*100):.1f}%", "is_header": False},
                {"row": 5, "col": 0, "value": "Nợ phải trả", "is_header": False},
                {"row": 5, "col": 1, "value": f"{liabilities_prev:,} tỷ", "is_header": False},
                {"row": 5, "col": 2, "value": f"{liabilities_curr:,} tỷ", "is_header": False},
                {"row": 5, "col": 3, "value": f"{((liabilities_curr-liabilities_prev)/liabilities_prev*100):.1f}%", "is_header": False}
            ]
        }
    
    def _generate_financial_ratios_table(self, table_id: str, year: int) -> Dict:
        """Generate financial ratios table."""
        prev_year = year - 1
        
        # Generate realistic ratios
        ros_prev = random.uniform(7, 12)
        ros_curr = ros_prev + random.uniform(-1, 2)
        
        roa_prev = random.uniform(3, 7)
        roa_curr = roa_prev + random.uniform(-0.5, 1.5)
        
        roe_prev = random.uniform(6, 12)
        roe_curr = roe_prev + random.uniform(-0.5, 2)
        
        debt_equity_prev = random.uniform(0.5, 0.8)
        debt_equity_curr = debt_equity_prev + random.uniform(-0.1, 0.1)
        
        return {
            "id": table_id,
            "caption": f"Bảng {table_id.replace('t', '')}: Các chỉ số tài chính chủ yếu",
            "col_headers": ["Chỉ số", f"Năm {prev_year}", f"Năm {year}", "Thay đổi"],
            "row_headers": [],
            "cells": [
                {"row": 0, "col": 0, "value": "Chỉ số", "is_header": True},
                {"row": 0, "col": 1, "value": f"Năm {prev_year}", "is_header": True},
                {"row": 0, "col": 2, "value": f"Năm {year}", "is_header": True},
                {"row": 0, "col": 3, "value": "Thay đổi", "is_header": True},
                {"row": 1, "col": 0, "value": "ROS (%)", "is_header": False},
                {"row": 1, "col": 1, "value": f"{ros_prev:.2f}%", "is_header": False},
                {"row": 1, "col": 2, "value": f"{ros_curr:.2f}%", "is_header": False},
                {"row": 1, "col": 3, "value": f"{ros_curr-ros_prev:+.2f}%", "is_header": False},
                {"row": 2, "col": 0, "value": "ROA (%)", "is_header": False},
                {"row": 2, "col": 1, "value": f"{roa_prev:.2f}%", "is_header": False},
                {"row": 2, "col": 2, "value": f"{roa_curr:.2f}%", "is_header": False},
                {"row": 2, "col": 3, "value": f"{roa_curr-roa_prev:+.2f}%", "is_header": False},
                {"row": 3, "col": 0, "value": "ROE (%)", "is_header": False},
                {"row": 3, "col": 1, "value": f"{roe_prev:.2f}%", "is_header": False},
                {"row": 3, "col": 2, "value": f"{roe_curr:.2f}%", "is_header": False},
                {"row": 3, "col": 3, "value": f"{roe_curr-roe_prev:+.2f}%", "is_header": False},
                {"row": 4, "col": 0, "value": "Nợ/Vốn CSH", "is_header": False},
                {"row": 4, "col": 1, "value": f"{debt_equity_prev:.2f}", "is_header": False},
                {"row": 4, "col": 2, "value": f"{debt_equity_curr:.2f}", "is_header": False},
                {"row": 4, "col": 3, "value": f"{debt_equity_curr-debt_equity_prev:+.2f}", "is_header": False}
            ]
        }
    
    def generate_dataset(
        self,
        num_documents: int = 20,
        output_dir: str = "data/synthetic_documents",
        start_index: int = 0
    ) -> List[str]:
        """
        Generate multiple synthetic documents.
        
        Args:
            num_documents: Number of documents to generate
            output_dir: Directory to save documents
            start_index: Starting index for document numbering (for resumption)
            
        Returns:
            List of generated document file paths
        """
        print(f"\n{'='*80}")
        print(f"Generating {num_documents} synthetic Vietnamese financial documents...")
        if start_index > 0:
            print(f"(Starting from index {start_index})")
        print(f"{'='*80}\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i in range(num_documents):
            doc_index = start_index + i
            # Random selection for diversity
            company = random.choice(self.COMPANIES)
            year = random.choice(self.YEARS)
            
            doc_id = f"doc_{company['name'].lower().replace(' ', '_')}_{year}_{doc_index:03d}"
            
            print(f"[{i+1}/{num_documents}]", end=" ")
            
            try:
                doc = self.generate_document(doc_id, company, year)
                
                # Save to file
                filename = f"{doc_id}.json"
                filepath = output_path / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
                
                generated_files.append(str(filepath))
                print(f"    ✓ Saved to {filename}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {len(generated_files)} documents")
        print(f"✓ Saved to: {output_dir}/")
        print(f"{'='*80}\n")
        
        return generated_files
