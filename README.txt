Updated GUI test fix for A2B ZIP (5), 17 Nov 2026.
Extract to project root, replacing tests/test_gui_backend.py.
Only changes test_partial_route_json_is_serialisable_by_gradio: it simulates missing sites 3685/4264 to keep partial-network test independent of model checkpoint coverage; preserves route map assertions from ZIP (5).
Run: python -m unittest discover -s tests -v
