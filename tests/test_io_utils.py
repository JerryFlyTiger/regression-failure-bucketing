"""ARCHITECTURE 5.2: normalize() must strip paths, then hex (0x-prefixed or
bare 8-digit hex), then decimal, before collapsing whitespace. The order
matters because bare hex digits are also valid decimal digits."""

from rfb.io_utils import normalize


def test_normalize_order_hex_before_decimal():
    s = "pc 8000f8e0 val 0x1f count 750 /a/b/c.sv"
    assert normalize(s) == "pc <HEX> val <HEX> count <N>"
