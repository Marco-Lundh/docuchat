"""Generates a sample employee handbook PDF for use in tests and manual demos.

Run directly to produce handbook.pdf in the project root:
    uv run src/tests/create_test_pdf.py
"""
import fitz

COMPANY = "Granit Software AB"

CONTENT = [
    (f"Employee Handbook – {COMPANY}", None),
    ("1. Welcome", f"""
Welcome to {COMPANY}! We are a software company founded in 2012
with offices in Stockholm and Uppsala. We currently employ 180
people and have an annual revenue of approximately 420 million
SEK. Our vision is to deliver reliable and maintainable systems
to the Swedish public sector.
"""),
    ("2. Working hours and flextime", """
Standard working hours are 40 hours per week. All employees are
covered by a flextime agreement. Core hours are Monday to Friday
09:00-15:00 and you must be available (in the office or remotely)
during this time. Flex time may be taken in full or half days
with approval from your line manager. Overtime is compensated at
1.5x hourly rate on weekdays and 2x on weekends.
"""),
    ("3. Vacation and leave", """
All employees are entitled to 30 vacation days per year.
The vacation year runs from 1 April to 31 March. Up to 10
vacation days per year may be carried over, with a maximum of
30 saved days. Parental leave is paid according to the Social
Insurance Agency rules plus a 10% salary supplement from the
company during the first 6 months. Unpaid leave for studies may
be granted for up to 12 months after 2 years of employment.
"""),
    ("4. Salary and benefits", """
Salary reviews take place once a year in April. Base salary is
set individually at the time of hiring. All employees receive a
wellness allowance of 5,000 SEK per year, which can be used for
gym, yoga, swimming and similar activities. Occupational pension
is paid at 4.5% of gross salary in addition to the statutory
contribution. All employees are offered a laptop of their choice
and may choose their preferred operating system.
"""),
    ("5. Remote work", """
The company operates a hybrid policy: at least 2 days per week
in the office are required. You choose which days you work from
home. Home office equipment (monitor, keyboard, headset) can be
borrowed from the IT department. More than 50% remote work in a
calendar month requires written approval from your manager and HR.
"""),
    ("6. Health and wellbeing", """
The company provides all employees with access to occupational
health services. Each employee is entitled to one wellbeing
consultation per quarter. We have an internal mentorship program
where senior employees are matched with junior colleagues.
Psychological support is included in the insurance. At least one
no-meeting day per week is strongly recommended for focus time.
"""),
    ("7. IT security", """
All employees must use the centrally provided password manager.
Two-factor authentication (2FA) is mandatory for all systems.
Work devices must be encrypted with FileVault (Mac) or BitLocker
(Windows). Incidents and suspected phishing must be reported
immediately to the security team. Personal USB drives are
prohibited on company computers.
"""),
    ("8. Contact information", """
HR department: hr@granitsoft.example | 08-100 200 00
IT support: it@granitsoft.example | Slack: #it-support
Security: security@granitsoft.example
CEO: ceo@granitsoft.example
Stockholm office: Example Street 1, 111 00 Stockholm
"""),
]


def create_pdf(output_path: str = "handbook.pdf") -> str:
    doc = fitz.open()
    font_size_title = 20
    font_size_heading = 14
    font_size_body = 11
    margin = 60
    page_width = 595
    page_height = 842
    text_width = page_width - 2 * margin

    page = doc.new_page(width=page_width, height=page_height)
    y = margin

    for i, (heading, body) in enumerate(CONTENT):
        if i == 0:
            page.insert_text(
                (margin, y), heading,
                fontsize=font_size_title, fontname="helv",
            )
            y += 36
            page.draw_line(
                (margin, y), (page_width - margin, y), width=1
            )
            y += 20
            continue

        if y > page_height - 150:
            page = doc.new_page(width=page_width, height=page_height)
            y = margin

        page.insert_text(
            (margin, y), heading,
            fontsize=font_size_heading, fontname="helv",
        )
        y += 22

        if body:
            lines = []
            for para in body.strip().split("\n"):
                para = para.strip()
                if not para:
                    continue
                words = para.split()
                line = ""
                for word in words:
                    test = f"{line} {word}".strip()
                    if fitz.get_text_length(
                        test, fontname="helv", fontsize=font_size_body
                    ) < text_width:
                        line = test
                    else:
                        lines.append(line)
                        line = word
                if line:
                    lines.append(line)
            lines.append("")

            for line in lines:
                if y > page_height - 60:
                    page = doc.new_page(width=page_width, height=page_height)
                    y = margin
                page.insert_text(
                    (margin, y), line,
                    fontsize=font_size_body, fontname="helv",
                )
                y += 16

        y += 10

    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    path = create_pdf()
    import fitz as _fitz
    page_count = _fitz.open(path).page_count
    print(f"Created {path} ({page_count} pages)")
