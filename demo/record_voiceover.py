"""Tek opp ei demo-video UTAN bileteksteregistreringar og med lange pausar,
slik at ein kan snakke over (voice-over). Skriv ut tidskodar for kvart steg.

Føresetnad: appen køyrer på http://127.0.0.1:7860  (python app.py)
Køyr:       python demo/record_voiceover.py
Resultat:   demo/demo-utan-tekst.mp4  (+ tidskodar i terminalen)
"""
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

from record_demo import H, OUT, URL, W, pick_station


def main():
    tmp = Path(tempfile.mkdtemp())
    cues = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(tmp),
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        t0 = time.monotonic()

        def cue(label):
            cues.append((time.monotonic() - t0, label))

        def hold(seconds):
            page.wait_for_timeout(int(seconds * 1000))

        page.goto(URL)
        page.wait_for_selector("text=Butikkrisiko per politistasjon")
        cue("1. Innleiing (fana Stasjon)")
        hold(20)

        cue("2. Delft blir valt")
        pick_station(page, "Delft")
        hold(17)

        cue("3. Kugompo blir valt")
        pick_station(page, "Kugompo")
        hold(12)

        cue("4. Fana 'Kva om?' opnast")
        page.get_by_role("tab", name="Kva om?").click()
        hold(6)
        cue("5. Khayelitsha med faktiske tal (30 innbrot, 12 ran)")
        pick_station(page, "Khayelitsha")
        page.get_by_role("button", name="Køyr modellen").click()
        hold(12)

        cue("6. Tala blir endra til 120 innbrot og 60 ran")
        page.get_by_label(re.compile("number input for Innbrot")).fill("120")
        page.get_by_label(re.compile("number input for Ran")).fill("60")
        page.get_by_role("button", name="Køyr modellen").click()
        hold(16)

        cue("7. Fana 'Høgast risiko'")
        page.get_by_role("tab", name="Høgast risiko").click()
        hold(10)

        cue("8. Fana 'Om modellen'")
        page.get_by_role("tab", name="Om modellen").click()
        hold(12)
        cue("9. Rullar ned til testresultat og avgrensingar")
        page.mouse.wheel(0, 500)
        hold(16)
        cue("SLUTT")

        video_path = Path(page.video.path())
        ctx.close()
        browser.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = OUT / "demo-utan-tekst.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mp4)],
        check=True, capture_output=True,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    print("Lagra", mp4, f"({mp4.stat().st_size / 1e6:.1f} MB)\n")
    for t, label in cues:
        print(f"{int(t // 60)}:{int(t % 60):02d}  {label}")


if __name__ == "__main__":
    main()
