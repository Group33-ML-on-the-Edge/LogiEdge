#!/usr/bin/env python3
"""Merge the individual task reports (reports/individual/) into phase1_report.pdf, phase2_report.pdf and final_report.pdf.

Each output gets a cover page with the contents and one bookmark per task. The grouping below is an ASSUMPTION based on
where the assignment text says each part belongs (C2's experiment goes in the Phase 2 report; A1, A2, C3 and D3 go in the
Final Report, with D3 as its Section 3). Edit GROUPS to regroup, then run:  python3 build_reports.py   (needs pypdf, reportlab)
"""
import os
import sys

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "individual")
TITLES = {
    "A1": ("Constraint analysis", "LogiEdge_Task_A1_Constraint_Analysis.pdf"),
    "A2": ("System architecture diagram", "LogiEdge_Task_A2_System_Architecture.pdf"),
    "B1": ("Constraint Triangle and hardware choice", "LogiEdge_Task_B1_Constraint_Triangle.pdf"),
    "B2": ("Arithmetic intensity and Roofline analysis", "LogiEdge_Task_B2_Roofline_Analysis.pdf"),
    "C1": ("Sensor simulator", "LogiEdge_Task_C1_Sensor_Simulator.pdf"),
    "C2": ("Preprocessing pipeline and the shifted-statistics experiment", "LogiEdge_Task_C2_Preprocessing_Pipeline.pdf"),
    "C3": ("Data fusion justification", "LogiEdge_Task_C3_Data_Fusion_Justification.pdf"),
    "D1": ("Dataset generation and model training", "LogiEdge_Task_D1_Model_Training.pdf"),
    "D2": ("Docker containerisation and OTA demo", "LogiEdge_Task_D2_Docker_OTA.pdf"),
    "D3": ("The 10-stage pipeline mapping", "LogiEdge_Task_D3_Pipeline_Mapping.pdf"),
    "E1": ("PSI drift monitoring", "LogiEdge_Task_E1_PSI_Drift_Monitoring.pdf"),
    "E2": ("Ansible deployment playbook", "LogiEdge_Task_E2_Ansible_Playbook.pdf"),
    "E3": ("OTA strategy selection", "LogiEdge_Task_E3_OTA_Strategy.pdf"),
    "F1": ("Three model variants", "LogiEdge_Task_F1_Model_Variants.pdf"),
    "F2": ("Five-metric benchmarking", "LogiEdge_Task_F2_Benchmarking.pdf"),
    "F3": ("Deployment recommendation", "LogiEdge_Task_F3_Deployment_Recommendation.pdf"),
}
GROUPS = {
    "phase1_report.pdf": ("LogiBridge Phase 1 Report", "Hardware choice and the sensor pipeline foundation", ["B1", "B2", "C1"]),
    "phase2_report.pdf": ("LogiBridge Phase 2 Report", "Preprocessing experiment, model training, deployment and monitoring", ["C2", "D1", "D2", "E1", "E2"]),
    "final_report.pdf": ("LogiBridge Final Report", "Architecture, design justification, pipeline mapping, OTA strategy and model optimisation",
                         ["A1", "A2", "D3", "C3", "E3", "F1", "F2", "F3"]),
}


def cover(path, title, subtitle, items, pages):
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    c.setFillColorRGB(0.043, 0.353, 0.502); c.rect(0, h - 70 * mm, w, 70 * mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 24); c.drawString(20 * mm, h - 35 * mm, title)
    c.setFont("Helvetica", 11); c.drawString(20 * mm, h - 46 * mm, "FreightBridge Logistics Pvt. Ltd. | LogiEdge Edge AI pilot")
    c.drawString(20 * mm, h - 54 * mm, subtitle)
    c.setFillColorRGB(0.1, 0.1, 0.1); c.setFont("Helvetica-Bold", 13); c.drawString(20 * mm, h - 90 * mm, "Contents")
    y = h - 102 * mm
    c.setFont("Helvetica", 10.5)
    for i, (key, text) in enumerate(items, start=1):
        c.drawString(24 * mm, y, f"Section {i}   Task {key}: {text}")
        c.drawRightString(w - 20 * mm, y, f"page {pages[key]}")
        y -= 8 * mm
    c.setFont("Helvetica-Oblique", 8.5); c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(20 * mm, 18 * mm, "Each section is a self-contained report with its own calculations, assumptions and open items.")
    c.save()


def build(name, title, subtitle, keys):
    readers = {k: PdfReader(os.path.join(SRC, TITLES[k][1])) for k in keys}
    pages, page = {}, 2                                   # page 1 is the cover
    for k in keys:
        pages[k] = page
        page += len(readers[k].pages)
    tmp = os.path.join(HERE, "_cover.pdf")
    cover(tmp, title, subtitle, [(k, TITLES[k][0]) for k in keys], pages)
    out = PdfWriter()
    for pg in PdfReader(tmp).pages:
        out.add_page(pg)
    for i, k in enumerate(keys, start=1):
        first = len(out.pages)
        for pg in readers[k].pages:
            out.add_page(pg)
        out.add_outline_item(f"Section {i}: Task {k} - {TITLES[k][0]}", first)
    out.add_metadata({"/Title": title, "/Author": "LogiBridge pilot team"})
    with open(os.path.join(HERE, name), "wb") as f:
        out.write(f)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return len(out.pages)


def main() -> int:
    for name, (title, subtitle, keys) in GROUPS.items():
        print(f"{name}: {build(name, title, subtitle, keys)} pages ({', '.join(keys)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
