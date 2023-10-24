import os
from datetime import date

from fpdf import FPDF


class PDF(FPDF):
    pass


# A4: 210 x 297 mm
pos_note = """According to the classification of the
EEG signal, there is a high chance you
will have a seizure, which may lead
to further health complications.

You should seek medical attention as soon
as possible. A professional doctor will
be able to provide further clarification
regarding the report.
"""

neg_note = """According to the classification of the
EEG signal, there is a low chance you
will have a seizure. At this time there
is no need for you to take further action.
"""


def createpdf(patient_data, idx, result, savepath):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=40)
    pdf.set_text_color(0, 0, 255)
    pdf.line(20, 30, 190, 30)

    # print the patient data
    pdf.cell(0, 20, txt=f"Patient Report / {idx}", align="C", ln=True)
    pdf.set_font("Arial", size=25)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(
        5,
        20,
        txt=f"""Name: {patient_data["first_name"]} {patient_data["last_name"]}""",
        ln=True,
    )
    pdf.cell(5, 20, txt=f"""Date of Birth: {patient_data["birth_date"]}""", ln=True)
    pdf.cell(5, 20, txt=f"""Email: {patient_data["email"]}""", ln=True)
    pdf.cell(
        5,
        20,
        txt=f"""Date of Diagnosis: {date.today().strftime("%d/%m/%Y")}""",
        ln=True,
    )

    pdf.image("tmp.png", w=180, h=100)
    os.remove("tmp.png")

    # Add doctor's note to pdf
    pdf.add_page()
    pdf.set_font("Arial", size=25)

    pdf.cell(5, 20, "Doctor's note:----", ln=True)
    note = pos_note if result else neg_note
    pdf.multi_cell(0, 10, note)

    pdf.output(savepath)


if __name__ == "__main__":
    createpdf(
        {
            "first_name": "e2we32",
            "last_name": "23e32",
            "email": "123",
            "birth_date": "01/12/1999",
        },
        1,
        1,
        "test.pdf",
    )
