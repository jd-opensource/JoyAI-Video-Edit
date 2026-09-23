import unittest
from types import SimpleNamespace

from xvideo.fp8_status import format_fp8_precision_status, fp8_precision_status


def _transformer(*blocks):
    return SimpleNamespace(double_blocks=list(blocks))


class Fp8PrecisionStatusTest(unittest.TestCase):
    def test_disabled_stream_is_bf16(self):
        status = fp8_precision_status(
            _transformer(SimpleNamespace(), SimpleNamespace()),
            {"img": False, "txt": False},
        )

        self.assertEqual(status["img"]["effective"], "bf16")
        self.assertEqual(status["txt"]["effective"], "bf16")

    def test_requested_stream_is_pending_before_lazy_conversion(self):
        status = fp8_precision_status(
            _transformer(SimpleNamespace(), SimpleNamespace()),
            {"img": True, "txt": True},
        )

        self.assertEqual(status["img"]["effective"], "pending")
        self.assertEqual(status["txt"]["effective"], "pending")
        self.assertIn("img=pending (requested=yes; 0/2 blocks converted lazily)",
                      format_fp8_precision_status(status))

    def test_reports_installed_and_partial_streams(self):
        status = fp8_precision_status(
            _transformer(
                SimpleNamespace(_fp8_img_installed=True, _fp8_txt_installed=True),
                SimpleNamespace(_fp8_img_installed=True),
            ),
            {"img": True, "txt": True},
        )

        self.assertEqual(status["img"]["effective"], "fp8")
        self.assertEqual(status["txt"]["effective"], "mixed")
        summary = format_fp8_precision_status(status)
        self.assertIn("img=fp8 (requested=yes; 2/2 blocks converted)", summary)
        self.assertIn(
            "txt=mixed (requested=yes; FP8 active in 1/2 blocks; conversion incomplete)",
            summary,
        )


if __name__ == "__main__":
    unittest.main()
