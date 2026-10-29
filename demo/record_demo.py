"""Tek opp ei demo-video av Gradio-appen med Playwright.

Føresetnad: appen køyrer på http://127.0.0.1:7860  (python app.py)
Køyr:       python demo/record_demo.py
Krev:       pip install playwright imageio-ffmpeg && python -m playwright install chromium
Resultat:   demo/demo.mp4
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = Path(__file__).resolve().parent
W, H = 1280, 1000


def caption(page, text):
    page.evaluate(
        """(t) => {
            let d = document.getElementById('demo-caption');
            if (!d) {
                d = document.createElement('div');
                d.id = 'demo-caption';
                d.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;'
                  + 'background:rgba(20,20,20,.88);color:#fff;font:600 22px sans-serif;'
                  + 'padding:14px 24px;text-align:center;';
                document.body.appendChild(d);
            }
            d.textContent = t;
        }""",
        text,
    )


def pick_station(page, name):
    box = page.get_by_label("Politistasjon").locator("visible=true").first
    box.click()
    box.press("Control+A")
    box.press_sequentially(name, delay=120)
    page.wait_for_timeout(800)
    page.locator(f'[role="option"][aria-label="{name}"]').first.click()
    page.wait_for_timeout(2500)


def main():
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(tmp),
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_selector("text=Butikkrisiko per politistasjon")
        page.wait_for_timeout(3000)

        caption(page, "Butikkrisiko per politistasjon: ML-modell trent på SAPS-data 2005–2026")
        page.wait_for_timeout(3500)

        caption(page, "Fane 1: Vel ein politistasjon og få prediksjon for neste sesong (2026/27)")
        page.wait_for_timeout(4000)
        pick_station(page, "Delft")
        caption(page, "Prediksjon, 80 %-intervall, risikonivå og rangering for Delft")
        page.wait_for_timeout(4500)
        pick_station(page, "Kugompo")
        caption(page, "Kugompo: ein stasjon med høgt risikonivå")
        page.wait_for_timeout(4500)

        page.get_by_role("tab", name="Kva om?").click()
        page.wait_for_timeout(1500)
        caption(page, "Fane 2: Kva om? Endre fjorårstala og sjå korleis modellen reagerer")
        page.wait_for_timeout(3500)
        pick_station(page, "Khayelitsha")
        page.get_by_role("button", name="Køyr modellen").click()
        caption(page, "Khayelitsha med fjorårets tal")
        page.wait_for_timeout(5000)
        caption(page, "Vi set inn 120 innbrot og 60 ran i staden, og køyrer modellen på nytt")
        page.get_by_label(re.compile("number input for Innbrot")).fill("120")
        page.get_by_label(re.compile("number input for Ran")).fill("60")
        page.get_by_role("button", name="Køyr modellen").click()
        page.wait_for_timeout(2500)
        caption(page, "Prediksjonen og risikonivået stig, så modellen reagerer på fjorårstala")
        page.wait_for_timeout(5500)

        page.get_by_role("tab", name="Høgast risiko").click()
        page.wait_for_timeout(1500)
        caption(page, "Fane 3: Dei 15 stasjonane med høgast predikert risiko")
        page.wait_for_timeout(5000)

        page.get_by_role("tab", name="Om modellen").click()
        page.wait_for_timeout(1500)
        caption(page, "Fane 4: Om modellen, testresultat (2025/26) og avgrensingar")
        page.wait_for_timeout(5000)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(4000)

        video_path = Path(page.video.path())
        ctx.close()
        browser.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = OUT / "demo.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mp4)],
        check=True, capture_output=True,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    print("Lagra", mp4, f"({mp4.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
