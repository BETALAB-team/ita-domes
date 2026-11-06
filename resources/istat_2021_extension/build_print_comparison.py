from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

root = Path.cwd()
output = root / "output/pdf"
output.mkdir(parents=True, exist_ok=True)
published = root / "output/temporal_extension_2013_2021_20261104"
revision = root / "output/article_method_2021_revision_20261104"
benchmark = pd.read_csv(published / "00_baseline/published_benchmarks.csv")
archetypes = pd.read_csv(root / "output/article_method_2021_20261104/02_probabilities/archetype_summary_2021.csv").set_index("archetype")
probability = revision / "02_probability_candidates"
energy = revision / "03_energy_candidates"
figures = revision / "04_figures"
paper = PdfReader("C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main.pdf")
figure6 = paper.pages[7].images[0].image
figure7 = paper.pages[8].images[0].image
with Image.open(published / "04_report/paper_figure_corrected/figures/published_figure8_2013.png") as src:
    assert src.size == (1521, 1900)
    figure8 = {"H": src.crop((0, 0, 1521, 867)), "E": src.crop((0, 977, 1521, 1822))}

width, height = A4
margin = 42
content = width - 2 * margin
ink = colors.black
rule = colors.black
for font, file in [("Roman", "times.ttf"), ("Bold", "timesbd.ttf"), ("Italic", "timesi.ttf")]:
    pdfmetrics.registerFont(TTFont(font, str(Path("C:/Windows/Fonts") / file)))
palette = ["#4169E1", "#50C878", "#FFBF00", "#D50032"]
labels = ["A1", "A2", "A3", "A4"]
terms = ["intercept", "age/100", "area/100", "age*area/10000", "age^2/10000", "area^2/10000"]
locations = ["North East", "North West", "Central", "South and Islands"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.linewidth": 0.65})


def historical(kind, group, metric):
    row = benchmark[(benchmark.kind == kind) & (benchmark.group == group) & (benchmark.metric == metric)]
    assert len(row) == 1, (kind, group, metric)
    return float(row.value.iloc[0])


def draw_text(pdf, text, x, y, font="Roman", size=11, align="left", color=ink):
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    {"left": pdf.drawString, "right": pdf.drawRightString, "center": pdf.drawCentredString}[align](x, y, str(text))


def page(pdf, year, title, page_no):
    draw_text(pdf, "Italian household archetypes", margin, 803, "Italic", 10)
    draw_text(pdf, year, width - margin, 803, "Bold", 12, "right")
    pdf.setStrokeColor(rule)
    pdf.setLineWidth(.45)
    pdf.line(margin, 793, width - margin, 793)
    draw_text(pdf, title, margin, 758, "Bold", 19)
    draw_text(pdf, "Khajedehi et al. (2025), doi:10.1016/j.enbuild.2025.115924" if year == 2013 else "ISTAT 2021 household survey",
              margin, 28, size=8)
    draw_text(pdf, f"{page_no} / 6", width - margin, 28, size=9, align="right")


def table(pdf, top, headers, rows, widths, row_height=25, size=10):
    assert abs(sum(widths) - content) < 0.1
    for i, row in enumerate([headers, *rows]):
        y = top - (i + 1) * row_height
        x = margin
        for j, (cell, cell_width) in enumerate(zip(row, widths, strict=True)):
            font = "Bold" if i == 0 else "Roman"
            assert pdf.stringWidth(str(cell), font, size) < cell_width - 5, (cell, cell_width)
            draw_text(pdf, cell, x + (8 if j == 0 else cell_width / 2), y + (row_height-size)/2 + 2,
                      font, size, "left" if j == 0 else "center")
            x += cell_width
    pdf.setStrokeColor(rule)
    pdf.setLineWidth(.75)
    pdf.line(margin, top, width - margin, top)
    pdf.setLineWidth(.35)
    pdf.line(margin, top - row_height, width - margin, top - row_height)
    pdf.setLineWidth(.75)
    pdf.line(margin, top - row_height * (len(rows) + 1), width - margin, top - row_height * (len(rows) + 1))


def picture(pdf, image, bottom, max_width, max_height):
    source = ImageReader(image)
    iw, ih = source.getSize()
    scale = min(max_width / iw, max_height / ih)
    pdf.drawImage(source, (width - iw * scale) / 2, bottom + (max_height - ih * scale) / 2,
                  iw * scale, ih * scale, mask="auto")


def plot_image(fig):
    buffer = BytesIO()
    fig.savefig(buffer, dpi=280, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def demographics(pdf, year):
    draw_text(pdf, "Table 1. Household profiles, mean (SD)", margin, 721, "Bold", 11)
    rows, shares = [], []
    for index, label in enumerate(labels, 1):
        if year == 2013:
            get = lambda metric: historical("demographic", label, metric)
            profile = [f"{get(name+'_mean'):.1f} ({get(name+'_sd'):.1f})" for name in ("min_age", "max_age", "number", "studio_ind")]
            household, people = get("household_share_pct"), get("people_share_pct")
        else:
            row = archetypes.loc[index]
            profile = [f"{row[name+'_mean']:.1f} ({row[name+'_sd']:.1f})" for name in ("min_age", "max_age", "number")]
            profile.append(f"{row.education_mean:.2f} ({row.education_sd:.2f})")
            household, people = row.household_share, row.person_share
        shares.append((household, people))
        rows.append([label, *profile, f"{household:.1f}%", f"{people:.1f}%"])
    table(pdf, 704, ["Cluster", "Min. age", "Max. age", "Members", "Education", "Households", "People"], rows,
          [56, 75, 75, 68, 86, 85, content-445], 32)
    tones = [.20, .43, .67, .88]
    bar_x = margin + 88
    bar_width = content - 102
    for col, (name, y) in enumerate([("Households", 393), ("People", 295)]):
        draw_text(pdf, name, margin, y+14, "Roman", 11)
        base = bar_x
        for row, tone in zip(shares, tones, strict=True):
            share = row[col]
            span = bar_width * share / 100
            pdf.setFillGray(tone)
            pdf.setStrokeGray(0)
            pdf.setLineWidth(.35)
            pdf.rect(base, y, span, 42, fill=1, stroke=1)
            draw_text(pdf, f"{share:.1f}%", base + span/2, y+17, "Roman", 10, "center",
                      colors.white if tone < .5 else ink)
            base += span
    for value in range(0, 101, 20):
        x = bar_x + bar_width * value / 100
        draw_text(pdf, f"{value}%", x, 272, size=9, align="center")
    for i, (tone, label) in enumerate(zip(tones, labels, strict=True)):
        x = bar_x + i * 82
        pdf.setFillGray(tone)
        pdf.rect(x, 230, 13, 10, fill=1, stroke=1)
        draw_text(pdf, label, x+18, 230, size=10)
    draw_text(pdf, "Fig. 1. Archetype shares among households and people.", margin, 202, size=10)
    draw_text(pdf, "Source: published Table 3 and Figure 5." if year == 2013 else "52,617 households. Shares are unweighted.",
              margin, 181, "Italic", 9)


def bracket(pdf, left, right, bottom, top):
    pdf.setStrokeColor(ink)
    pdf.setLineWidth(.75)
    path = pdf.beginPath()
    path.moveTo(left+5, top)
    path.lineTo(left, top)
    path.lineTo(left, bottom)
    path.lineTo(left+5, bottom)
    path.moveTo(right-5, top)
    path.lineTo(right, top)
    path.lineTo(right, bottom)
    path.lineTo(right-5, bottom)
    pdf.drawPath(path)


def probability_matrix(pdf, coefficients, year):
    center_y = 654
    left = (width - 463) / 2
    bracket(pdf, left, left+37, center_y-34, center_y+34)
    bracket(pdf, left+62, left+363, center_y-34, center_y+34)
    bracket(pdf, left+390, left+463, center_y-49, center_y+49)
    draw_text(pdf, "=", left+49, center_y-4, "Roman", 13, "center")
    draw_text(pdf, "×", left+376, center_y-4, "Roman", 13, "center")
    for i, row in enumerate(coefficients):
        y = center_y + 20 - i*16
        draw_text(pdf, "p", left+13, y, "Italic", 12)
        draw_text(pdf, i+1, left+19, y-2, "Roman", 8)
        for j, coefficient in enumerate(row):
            precision = 3 if year == 2013 else 5
            value = f"{coefficient:.{precision}f}".replace("-", "−")
            draw_text(pdf, value, left+108+j*49, y, "Roman", 11, "right")
    for i, entry in enumerate(("1", "a / 100", "s / 100", "as / 10⁴", "a² / 10⁴", "s² / 10⁴")):
        draw_text(pdf, entry, left+426.5, center_y+36-i*16, "Roman", 11, "center")


def probability_fit(pdf, year):
    draw_text(pdf, "Fitted archetype probabilities", margin, 722, "Bold", 11)
    if year == 2013:
        coefficients = [[historical("polynomial_Eq3", label, term) for term in terms] for label in labels]
        rows = [[title, *[f"{historical('polynomial_fit', label, key):.{digits}f}" for label in labels]]
                for title, key, digits in [("Adjusted R²", "adjusted_R2", 2), ("RMSE", "RMSE", 3)]]
        chart = figure7
    else:
        coeff = pd.read_csv(probability / "recommended_full_quadratic_matrix.csv")
        coefficients = coeff[terms].to_numpy()
        fit = pd.read_csv(probability / "recommended_apparent_fit_statistics.csv")
        rows = [["Adjusted R²", *[f"{v:.3f}" for v in fit.adjusted_R2]], ["RMSE", *[f"{v:.5f}" for v in fit.RMSE]]]
        data = pd.read_csv(figures / "figure_6_7_node_data.csv")
        fig, ax = plt.subplots(figsize=(4.6, 4.4), layout="constrained")
        for i, hue in enumerate(palette, 1):
            ax.scatter(data[f"p{i}_obs"], data[f"p{i}_fit"], s=13, marker="x", linewidth=.75, color=hue, label=f"Cluster {i}")
        ax.plot([0, 1], [0, 1], color="#8F969D", linewidth=.7, linestyle="--", label="y = x")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Actual Proportion", ylabel="Predicted Probability", aspect="equal")
        ax.grid(color="#D7DCE0", linewidth=.5)
        ax.legend(loc="upper left", fontsize=8)
        chart = plot_image(fig)
    probability_matrix(pdf, coefficients, year)
    draw_text(pdf, "a = dwelling age (years); s = floor area (m²).", margin, 588, "Italic", 10)
    draw_text(pdf, "Coefficients from Eq. (3). Table 4 lists 0.02 for the A4 age² term." if year == 2013 else
              "Full quadratic fit to 125 windows.", margin, 573, "Italic", 9)
    draw_text(pdf, "Table 2. Regression fit statistics", margin, 548, "Bold", 11)
    table(pdf, 538, ["Statistic", *labels], rows, [183, *[(content-183)/4]*4], 19)
    picture(pdf, chart, 83, 390, 378)
    draw_text(pdf, "Fig. 3. Predicted probabilities and observed proportions." + (" Source: published Figure 7." if year == 2013 else ""),
              margin, 65, size=10)


def energy_page(pdf, year, prefix):
    heating = prefix == "H"
    outcome = "Heating system on-time (hours/day)" if heating else "Annual appliance consumption (kWh/year)"
    draw_text(pdf, ("Table 3. " if heating else "Table 4. ") + outcome, margin, 722, "Bold", 11)
    if year == 2013:
        data = benchmark[benchmark.kind == ("heating" if heating else "electricity")]
        rows = [[r.group, f"{r.value:.2f}" if heating else f"{r.value:,.0f}",
                 f"{r.sd:.2f}" if heating else f"{r.sd:,.0f}"] for r in data.itertuples()]
        chart = figure8[prefix]
    else:
        data = pd.read_csv(energy / f"{prefix}_selected_group_summary.csv")
        rows = [[r.group, f"{r.mean:.2f}" if heating else f"{r.mean:,.0f}",
                 f"{r.sd:.2f}" if heating else f"{r.sd:,.0f}"] for r in data.itertuples()]
        hh = pd.read_csv(energy / "compact_assignments_with_outcomes.csv")
        group_col = f"{prefix}_compact_group"
        outcome_col = "heating_daily_hours" if heating else "electricity_annual_kwh"
        fig, ax = plt.subplots(figsize=(7.3, 4.15), layout="constrained")
        style = dict(patch_artist=True, showfliers=False, manage_ticks=False,
                     medianprops={"color": "#343B42", "linewidth": .8},
                     whiskerprops={"color": "#555D65", "linewidth": .7},
                     capprops={"color": "#555D65", "linewidth": .7})
        for i, group in enumerate(data.group, 1):
            subset = hh[hh[group_col] == group]
            strata = [(0, "#D8A779", subset)] if heating else [
                (-.17, "#A9BFD5", subset[subset.cooling_system_use == "absent_or_unused"]),
                (.17, "#23354E", subset[subset.cooling_system_use == "present_or_used"])]
            for offset, hue, stratum in strata:
                ax.boxplot(stratum[outcome_col].dropna(), positions=[i+offset], widths=.58 if heating else .34,
                           boxprops={"facecolor": hue, "edgecolor": "#343B42", "linewidth": .7}, **style)
        ax.set_xticks(range(1, len(rows)+1), data.group)
        ax.set(xlim=(.5, len(rows)+.5), ylim=(0, 24 if heating else 5000),
               ylabel="Daily On-Time (hours)" if heating else "Annual Appliance Consumption (kWh)")
        ax.grid(axis="y", color="#CACFD4", linestyle="--", linewidth=.55)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor="#D8A779", label="All eligible households")] if heating else [
            Patch(facecolor="#A9BFD5", label="Cooling System Not Used"),
            Patch(facecolor="#23354E", label="Cooling System Used")]
        ax.legend(handles=handles, loc="upper left", fontsize=9)
        chart = plot_image(fig)
    table(pdf, 705, ["Group", "Mean", "SD"], rows, [145, (content-145)/2, (content-145)/2], 22)
    picture(pdf, chart, 176, content, 320)
    draw_text(pdf, ("Fig. 4. Heating system on-time" if heating else "Fig. 5. Annual appliance electricity") +
              " by group.", margin, 154, size=10)
    note = ("Table 6 and Figure 8(a)" if heating else "Table 6 and Figure 8(b)") if year == 2013 else (
        "Compact H1-H4 · pooled boxes · 48,690 households" if heating else "Compact E1-E5 · 52,617 households")
    draw_text(pdf, note, margin, 137, "Italic", 9)
    draw_text(pdf, "Group labels indicate rank within each year.", margin, 122, "Italic", 9)


def appendix(pdf, year):
    if year == 2013:
        data = pd.read_csv(root / "output/claim_checked_article/00_context/archetype_extension/published_appendix_A_group_lookup.csv")
        dwellings = ["Apartment in Small Building", "Apartment in Medium Building", "Apartment in Large Building", "Single Family House", "Multi Family House"]
        display = ["Apartment in small building", "Apartment in medium building", "Apartment in large building", "Single-family house", "Multi-family house"]
        data = data.set_index(["published_location", "published_dwelling_type", "user_archetype_id"])
        assert len(data) == 80 and data.index.is_unique
        groups = [[data.loc[(loc, dw, a), outcome] for outcome in ("heating_group", "electricity_group") for a in range(1, 5)]
                  for loc in locations for dw in dwellings]
    else:
        data = pd.read_csv(energy / "appendix_A_assignment_2021_wide_flat.csv")
        assert data.Location.tolist() == [loc for loc in locations for _ in range(5)]
        display = ["Apartment on ground, semi-\nbasement or raised floor", "Apartment on intermediate floor", "Apartment on top floor", "Single-family house", "Multi-family house"]
        groups = data.iloc[:, 2:].values.tolist()
    assert len(groups) == 20 and all(len(row) == 8 for row in groups)
    location_width, dwelling_width = 67, 150
    group_width = (content-location_width-dwelling_width)/8
    group_x = margin + location_width + dwelling_width
    top, row_height = 686, 27
    draw_text(pdf, "Table A.1. Energy group by location, dwelling type, and archetype", margin, 736, "Bold", 11)
    pdf.setStrokeColor(rule)
    pdf.setLineWidth(.75)
    pdf.line(margin, top+40, width-margin, top+40)
    draw_text(pdf, "Location", margin+8, top+13, "Bold", 10)
    draw_text(pdf, "Dwelling type", margin+location_width+8, top+13, "Bold", 10)
    for j, name in enumerate(("Heating", "Electricity")):
        draw_text(pdf, name, group_x+(2+4*j)*group_width, top+25, "Bold", 10, "center")
    for j in range(8):
        draw_text(pdf, labels[j % 4], group_x+(j+.5)*group_width, top+9, "Bold", 10, "center")
    pdf.setLineWidth(.35)
    pdf.line(margin, top, width-margin, top)
    for i, entries in enumerate(groups):
        y = top-(i+1)*row_height
        if i % 5 == 0:
            loc_lines = locations[i//5].replace("South and Islands", "South and\nIslands").splitlines()
            for j, line in enumerate(loc_lines):
                draw_text(pdf, line, margin+7, y-2*row_height+12-11*j, "Bold", 9.5)
        name = display[i % 5].splitlines()
        for j, line in enumerate(name):
            draw_text(pdf, line, margin+location_width+7, y+(9 if len(name)==1 else 15)-10*j, size=10)
        for j, group in enumerate(entries):
            assert group[0] == ("H" if j < 4 else "E")
            x = group_x + j*group_width
            draw_text(pdf, group, x+group_width/2, y+9, "Roman", 10, "center")
        if i % 5 == 4:
            pdf.setStrokeColor(rule)
            pdf.setLineWidth(.75 if i == 19 else .35)
            pdf.line(margin, y, width-margin, y)
    draw_text(pdf, "Published Appendix A · apartment classes by building size" if year == 2013 else
              "Compact H1-H4 / E1-E5 · apartment classes by floor position", margin, 118, size=9)


for year in (2013, 2021):
    filename = output / f"Italian_archetypes_{year}_print_comparison.pdf"
    pdf = canvas.Canvas(str(filename), pagesize=A4, pageCompression=1)
    pdf.setTitle(f"Italian household archetypes {year} | paired print comparison")
    page(pdf, year, "Household archetypes", 1)
    demographics(pdf, year)
    pdf.showPage()
    page(pdf, year, "Dwelling area and archetype probability", 2)
    picture(pdf, figure6 if year == 2013 else str(figures / "dwelling_probability_profiles_2021.png"), 155, content, 580)
    draw_text(pdf, "Fig. 2. Archetype probability by dwelling area and construction period.", margin, 146, size=10)
    draw_text(pdf, "Source: published Figure 6. Probability scales vary by panel." if year == 2013 else
              "Shared probability scale across panels.", margin, 131, "Italic", 9)
    pdf.showPage()
    page(pdf, year, "Probability model fit", 3)
    probability_fit(pdf, year)
    pdf.showPage()
    page(pdf, year, "Heating groups", 4)
    energy_page(pdf, year, "H")
    pdf.showPage()
    page(pdf, year, "Appliance electricity groups", 5)
    energy_page(pdf, year, "E")
    pdf.showPage()
    page(pdf, year, "Appendix A. Group assignments", 6)
    appendix(pdf, year)
    pdf.showPage()
    pdf.save()
    assert len(PdfReader(filename).pages) == 6
    print(filename)
