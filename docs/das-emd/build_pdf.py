#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("/mnt/d/dev/codex/DAS_EMD_Wasserstein1_Giai_thich.pdf")

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#008C87")
TEAL_DARK = colors.HexColor("#006F6B")
INK = colors.HexColor("#25313C")
MUTED = colors.HexColor("#5D6B78")
PALE = colors.HexColor("#EEF7F6")
PALE_BLUE = colors.HexColor("#EEF3F8")
LINE = colors.HexColor("#CFD9E2")
WHITE = colors.white


FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVBI", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVMono", str(FONT_DIR / "DejaVuSansMono.ttf")))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="DVB",
            fontSize=23,
            leading=29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="DV",
            fontSize=11.2,
            leading=16,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="DVB",
            fontSize=16,
            leading=21,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="DVB",
            fontSize=12.2,
            leading=16,
            textColor=TEAL_DARK,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="DV",
            fontSize=10.25,
            leading=15.3,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "body_left": ParagraphStyle(
            "BodyLeft",
            parent=base["BodyText"],
            fontName="DV",
            fontSize=10.25,
            leading=15.3,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="DV",
            fontSize=8.6,
            leading=12.2,
            textColor=MUTED,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["BodyText"],
            fontName="DV",
            fontSize=11.3,
            leading=17,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="DVI",
            fontSize=10.1,
            leading=15,
            textColor=NAVY,
            leftIndent=7,
            rightIndent=7,
            alignment=TA_LEFT,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle",
            parent=base["BodyText"],
            fontName="DVB",
            fontSize=10.3,
            leading=14,
            textColor=TEAL_DARK,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="DV",
            fontSize=9.7,
            leading=14.2,
            textColor=INK,
        ),
        "card_title": ParagraphStyle(
            "CardTitle",
            parent=base["BodyText"],
            fontName="DVB",
            fontSize=9.3,
            leading=12,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="DVB",
            fontSize=9.3,
            leading=12,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "card": ParagraphStyle(
            "Card",
            parent=base["BodyText"],
            fontName="DV",
            fontSize=8.2,
            leading=11.4,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


S = make_styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def equation(html: str) -> Table:
    box = Table([[P(html, "equation")]], colWidths=[168 * mm], hAlign="CENTER")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.65, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return box


def callout(title: str, text: str, color=PALE) -> Table:
    content = [P(title, "callout_title"), P(text, "callout")]
    box = Table([[content]], colWidths=[168 * mm], hAlign="CENTER")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.75, TEAL),
                ("LINEBEFORE", (0, 0), (0, -1), 3.2, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def bullets(items: list[str], numbered: bool = False) -> ListFlowable:
    kwargs = {
        "bulletType": "1" if numbered else "bullet",
        "leftIndent": 19,
        "bulletFontName": "DVB" if numbered else "DV",
        "bulletFontSize": 9,
        "bulletColor": TEAL_DARK,
        "spaceBefore": 1,
        "spaceAfter": 5,
    }
    if numbered:
        kwargs["start"] = "1"
    return ListFlowable(
        [ListItem(P(item, "body_left"), leftIndent=4) for item in items],
        **kwargs,
    )


def section(number: str, title: str) -> KeepTogether:
    return KeepTogether(
        [
            Spacer(1, 4),
            HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=5),
            P(f"{number}. {title}", "h1"),
        ]
    )


def process_diagram() -> Table:
    data = [
        [
            [P("Nhiễu ban đầu", "card_title"), P("x<sub>T</sub> ~ N(0, I)", "card")],
            P("→", "equation"),
            [P("Reverse diffusion + SMC", "card_title"), P("weight, ESS, resample qua từng bước", "card")],
            P("→", "equation"),
            [P("Terminal samples", "card_title"), P("x<sub>0</sub> trong data space 2D", "card")],
        ]
    ]
    t = Table(data, colWidths=[46 * mm, 10 * mm, 57 * mm, 10 * mm, 45 * mm], hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), PALE_BLUE),
                ("BACKGROUND", (2, 0), (2, 0), PALE),
                ("BACKGROUND", (4, 0), (4, 0), PALE_BLUE),
                ("BOX", (0, 0), (0, 0), 0.7, LINE),
                ("BOX", (2, 0), (2, 0), 0.7, TEAL),
                ("BOX", (4, 0), (4, 0), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return t


def ot_example_table() -> Table:
    data = [
        [P("Tình huống", "table_header"), P("Phân bổ X", "table_header"), P("Hệ quả EMD", "table_header")],
        [P("DAS đúng", "callout_title"), P("50% trái, 50% phải", "callout"), P("Chỉ cần chuyển mass quãng ngắn → thấp", "callout")],
        [P("Mode collapse", "callout_title"), P("95% trái, 5% phải", "callout"), P("Khoảng 45% mass phải đi xa → cao", "callout")],
        [P("Mode lệch vị trí", "callout_title"), P("Tỷ lệ 50/50 nhưng tâm cụm sai", "callout"), P("Hầu hết mass vẫn phải dịch chuyển → cao", "callout")],
    ]
    t = Table(data, colWidths=[38 * mm, 51 * mm, 79 * mm], repeatRows=1, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
                ("GRID", (0, 0), (-1, -1), 0.45, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, height - 15 * mm, width - 20 * mm, height - 15 * mm)
        canvas.setFont("DV", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, height - 11.5 * mm, "DAS toy GMM • EMD / Wasserstein-1")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
    canvas.setFont("DV", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 9.5 * mm, "Ghi chú kỹ thuật • 13/08/2026")
    canvas.drawRightString(width - 20 * mm, 9.5 * mm, f"Trang {doc.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=21 * mm,
        bottomMargin=20 * mm,
        title="DAS toy GMM: EMD/Wasserstein-1 và ý nghĩa của X, Y, u_X, u_Y, pi",
        author="Codex",
        subject="Giải thích chi tiết metric EMD trong thí nghiệm toy GMM của DAS",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])

    story = []
    story += [
        Spacer(1, 7 * mm),
        P("DAS toy GMM", "title"),
        P("EMD / Wasserstein-1 và ý nghĩa chính xác của X, Y, u<sub>X</sub>, u<sub>Y</sub>, π", "title"),
        P(
            "Ghi chú này giải thích EMD trong <b>Figure 1 toy GMM của DAS</b>. "
            "Ở đây các sample là điểm hai chiều; chúng chưa phải ảnh.",
            "subtitle",
        ),
        callout(
            "Trả lời ngắn gọn",
            "<b>X</b> là các terminal sample x<sub>0</sub> do DAS/SMC sinh sau toàn bộ reverse process; "
            "<b>Y</b> là reference sample từ target reward-tilted distribution, không phải ảnh thật; "
            "<b>u<sub>X</sub>, u<sub>Y</sub></b> là các vector trọng số xác suất của empirical samples; "
            "<b>π</b> là kế hoạch vận chuyển mass tối ưu giữa hai tập.",
        ),
        Spacer(1, 9),
        process_diagram(),
        Spacer(1, 10),
        P(
            "Câu hỏi mà EMD trả lời: <i>Cần vận chuyển trung bình bao xa để biến empirical distribution "
            "do DAS sinh ra thành empirical target distribution?</i>",
            "quote",
        ),
    ]

    story += [section("1", "X là nhiễu hay output cuối?")]
    story += [
        P(
            "<b>X là output cuối</b> của quá trình reverse diffusion/SMC, không phải Gaussian noise ban đầu "
            "x<sub>T</sub>.",
        ),
        equation("<i>X</i> = {x<sub>0</sub><super>(1)</super>, …, x<sub>0</sub><super>(N)</super>}."),
        Spacer(1, 6),
        equation("x<sub>T</sub> ~ N(0, I) → x<sub>T-1</sub> → … → x<sub>1</sub> → x<sub>0</sub>."),
        Spacer(1, 6),
        bullets(
            [
                "<b>x<sub>T</sub></b>: nhiễu ban đầu.",
                "<b>x<sub>t</sub></b>: particle vẫn còn nhiễu ở bước trung gian.",
                "<b>x<sub>0</sub></b>: sample hoàn chỉnh trong data space.",
                "Trong toy GMM, mỗi x<sub>0</sub> = (x<sub>1</sub>, x<sub>2</sub>) chỉ là một điểm 2D.",
                "Trong text-to-image diffusion, x<sub>0</sub> tương ứng ảnh hoặc latent ảnh đã denoise.",
            ]
        ),
        P("Ở mỗi reverse step, DAS duy trì nhiều particles và thực hiện:", "body_left"),
        bullets(
            [
                "Dùng pretrained diffusion transition làm proposal.",
                "Ước lượng clean sample x̂<sub>0</sub>(x<sub>t</sub>).",
                "Đánh giá reward trên clean estimate.",
                "Cập nhật importance weights.",
                "Resample nếu effective sample size (ESS) quá thấp.",
                "Tiếp tục denoise.",
            ],
            numbered=True,
        ),
        callout(
            "Step nào được dùng để tính EMD?",
            "EMD trong Figure 1 <b>không</b> được tính trên noisy particles trung gian. Nó dùng "
            "<font name='DVMono'>smc_samples</font>, tức các terminal samples sau khi hoàn tất toàn bộ reverse process. "
            "Reproduction có khoảng 1.000 terminal samples, sau đó notebook rút 500 điểm để tính paper-style EMD.",
            PALE_BLUE,
        ),
    ]

    story += [section("2", "Y có phải ảnh thật không?")]
    story += [
        P("Không. Trong toy GMM, <b>Y</b> là các reference samples từ target reward-tilted distribution:"),
        equation("<i>Y</i> = {y<sub>1</sub>, …, y<sub>M</sub>},"),
        Spacer(1, 5),
        equation(
            "p<sub>tar</sub>(x) = (1/Z) p<sub>pre</sub>(x) "
            "exp(r(x)/α)."
        ),
        Spacer(1, 6),
        bullets(
            [
                "<b>p<sub>pre</sub></b>: phân phối GMM mà diffusion model ban đầu đã học.",
                "<b>r(x)</b>: reward.",
                "<b>α</b>: KL/reward temperature.",
                "<b>Z</b>: normalization constant.",
            ]
        ),
        P("Notebook tạo <font name='DVMono'>target_samples</font> theo bốn bước:", "body_left"),
        bullets(
            [
                "Tạo một lưới dày trong không gian 2D.",
                "Tính p<sub>pre</sub>(x) exp(r(x)/α) tại từng điểm lưới.",
                "Chuẩn hóa các giá trị thành xác suất.",
                "Sample 10.000 điểm từ phân phối lưới đã chuẩn hóa.",
            ],
            numbered=True,
        ),
        callout(
            "Vai trò của hai tập",
            "<b>X</b>: những gì DAS/SMC thực sự sinh ra. &nbsp;&nbsp; "
            "<b>Y</b>: những gì DAS được kỳ vọng phải sinh ra. &nbsp;&nbsp; "
            "EMD đo khoảng cách distributional giữa hai tập này.",
        ),
        P("<b>Trong text-to-image thì sao?</b>", "h2"),
        equation("p<sub>tar</sub>(x | c) ∝ p<sub>pre</sub>(x | c) exp(r(x,c)/α)."),
        Spacer(1, 6),
        P(
            "Trong text-to-image, ta thường không thể lấy mẫu chính xác từ target trên để tạo một tập Y. "
            "Vì vậy không đơn giản lấy ảnh thật làm Y rồi tính EMD trong pixel space. Các thí nghiệm thường dùng "
            "PickScore, HPS v2, ImageReward, Aesthetic, diversity metrics và human evaluation. EMD X-vs-Y ở đây "
            "đặc biệt phù hợp với toy GMM vì target distribution có thể được dựng gần như trực tiếp."
        ),
    ]

    story += [section("3", "u_X và u_Y là gì?")]
    story += [
        P("Từ hai tập samples, ta tạo hai empirical distributions:"),
        equation("P̂<sub>X</sub> = Σ<sub>i=1</sub><super>N</super> u<sub>X,i</sub> δ<sub>xᵢ</sub>,"),
        Spacer(1, 5),
        equation("P̂<sub>Y</sub> = Σ<sub>j=1</sub><super>M</super> u<sub>Y,j</sub> δ<sub>yⱼ</sub>."),
        Spacer(1, 6),
        bullets(
            [
                "<b>δ<sub>xᵢ</sub></b>: một point mass đặt tại sample x<sub>i</sub>.",
                "<b>u<sub>X,i</sub></b>: lượng probability mass mà sample x<sub>i</sub> đại diện.",
                "<b>u<sub>Y,j</sub></b>: lượng probability mass mà sample y<sub>j</sub> đại diện.",
            ]
        ),
        P("Nếu mọi sample có trọng số như nhau:"),
        equation("u<sub>X,i</sub> = 1/N, &nbsp;&nbsp; u<sub>Y,j</sub> = 1/M,"),
        Spacer(1, 5),
        equation("u<sub>X</sub> = (1/N, …, 1/N), &nbsp;&nbsp; u<sub>Y</sub> = (1/M, …, 1/M)."),
        Spacer(1, 6),
        callout(
            "“Uniform mass” không có nghĩa là samples phân bố đều trong không gian",
            "Nó chỉ có nghĩa là mỗi điểm quan sát mang cùng một trọng số xác suất. Nếu X có bốn điểm thì "
            "u<sub>X</sub> = (0.25, 0.25, 0.25, 0.25). Nếu ba điểm cùng nằm gần một mode, mode đó tự động "
            "mang khoảng 0.75 empirical mass. Vì thế uniform sample weights vẫn phản ánh đúng tần suất mode.",
            PALE_BLUE,
        ),
    ]

    story += [section("4", "π là gì?")]
    story += [
        P(
            "<b>π</b> là transport plan, một ma trận không âm kích thước N × M. Phần tử π<sub>ij</sub> "
            "cho biết chuyển bao nhiêu mass từ source sample x<sub>i</sub> sang target sample y<sub>j</sub>."
        ),
        equation("π ∈ R<sub>+</sub><super>N×M</super>."),
        Spacer(1, 5),
        P("Transport plan phải thỏa hai điều kiện biên:"),
        equation("Σ<sub>j</sub> π<sub>ij</sub> = u<sub>X,i</sub> &nbsp; (toàn bộ mass tại x<sub>i</sub> được chuyển đi),"),
        Spacer(1, 5),
        equation("Σ<sub>i</sub> π<sub>ij</sub> = u<sub>Y,j</sub> &nbsp; (y<sub>j</sub> nhận đúng target mass)."),
        Spacer(1, 6),
        P("Tập tất cả transport plan hợp lệ được ký hiệu Π(u<sub>X</sub>, u<sub>Y</sub>). EMD chọn plan có tổng chi phí nhỏ nhất:"),
        equation(
            "Ŵ<sub>1</sub>(X,Y) = min<sub>π ∈ Π(u<sub>X</sub>,u<sub>Y</sub>)</sub> "
            "Σ<sub>i,j</sub> π<sub>ij</sub> ||x<sub>i</sub> - y<sub>j</sub>||<sub>2</sub>."
        ),
        Spacer(1, 5),
        equation("c<sub>ij</sub> = ||x<sub>i</sub> - y<sub>j</sub>||<sub>2</sub>."),
        Spacer(1, 6),
        P(
            "Chi phí c<sub>ij</sub> là khoảng cách Euclidean để chuyển một đơn vị mass từ x<sub>i</sub> tới y<sub>j</sub>. "
            "Vì vậy EMD vừa nhạy với vị trí các mode, vừa nhạy với tỷ lệ mass giữa chúng."
        ),
    ]

    story += [section("5", "Ví dụ trực giác")]
    story += [
        P("Giả sử target Y có hai mode: 50% bên trái và 50% bên phải."),
        ot_example_table(),
        Spacer(1, 7),
        P(
            "Mode coverage có thể vẫn báo 2/2 trong trường hợp 95%/5%, vì cả hai mode đều có sample. "
            "EMD phát hiện vấn đề vì optimal transport buộc phải chuyển một lượng mass lớn qua khoảng cách xa."
        ),
        callout(
            "EMD đo đồng thời",
            "Vị trí mode • tỷ lệ mass giữa các mode • spurious samples • khoảng cách mà lượng mass sai phải di chuyển.",
        ),
    ]

    story += [section("6", "Tại sao không giữ importance weights cuối của SMC?")]
    story += [
        P("Trong generic importance sampling, ta có thể giữ một empirical measure có trọng số:"),
        equation("P̂<sub>X</sub> = Σ<sub>i</sub> w<sub>i</sub> δ<sub>xᵢ</sub>."),
        Spacer(1, 6),
        P(
            "Nhưng DAS thực hiện resampling để biến particle population thành một tập gần target. Sau terminal resampling, "
            "paper đánh giá chính tập generated samples như các samples thông thường và đặt trọng số bằng nhau:"
        ),
        equation("u<sub>X,i</sub> = 1/N."),
        Spacer(1, 6),
        callout(
            "Câu hỏi đánh giá phù hợp với generation",
            "<i>Nếu tôi lấy ngẫu nhiên một output trong tập DAS đã sinh, distribution của output đó có gần target không?</i><br/><br/>"
            "Khác với câu hỏi: <i>Nếu giữ thêm một bộ importance weights bên ngoài, liệu tôi có thể sửa lại distribution của tập output không?</i>",
            PALE_BLUE,
        ),
    ]

    story += [section("7", "Tóm tắt chính xác cho Figure 1 DAS")]
    story += [
        process_diagram(),
        Spacer(1, 8),
        P("Sau khi reverse process hoàn tất:"),
        equation("X = {x<sub>0</sub><super>(i)</super>}<sub>i=1</sub><super>500</super>"),
        Spacer(1, 5),
        equation("Y = {y<sub>j</sub>}<sub>j=1</sub><super>500</super>, &nbsp;&nbsp; y<sub>j</sub> ~ p<sub>tar</sub>,"),
        Spacer(1, 5),
        equation("u<sub>X</sub> = (1/500, …, 1/500), &nbsp;&nbsp; u<sub>Y</sub> = (1/500, …, 1/500)."),
        Spacer(1, 9),
        callout(
            "Kết luận",
            "EMD càng thấp thì terminal distribution của DAS càng gần target reward-tilted distribution. "
            "Nó không so nhiễu trung gian với ảnh thật; nó so <b>output cuối của SMC</b> với <b>reference samples từ target distribution</b>.",
        ),
        Spacer(1, 12),
        P(
            "Nguồn nội dung: phần giải thích đã chọn trong task nghiên cứu DAS/SMC. Tài liệu này chỉ chuyển thể và dàn trang; "
            "không thêm kết quả thực nghiệm mới.",
            "small",
        ),
    ]

    doc.build(story)


if __name__ == "__main__":
    build()
