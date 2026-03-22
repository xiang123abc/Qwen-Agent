from qwen_agent.kernel_patch.ordering_constraints import detect_ordering_constraints


def test_detect_ordering_constraints_flags_declaration_reorder():
    diff_text = """diff --git a/foo.c b/foo.c
@@
-\tstruct usb_device *usb_dev = hid_to_usb_dev(hdev);
-\tunsigned char *data = kmalloc(8, GFP_KERNEL);
+\tstruct usb_device *usb_dev;
+\tusb_dev = hid_to_usb_dev(hdev);
"""
    categories = detect_ordering_constraints(diff_text)
    assert 'statement_order_mismatch' in categories
    assert 'declaration_order_mismatch' in categories
