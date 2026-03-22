from qwen_agent.kernel_patch.cases import iter_patch_cases


def test_iter_patch_cases_skips_invalid_and_deduplicates():
    lines = [
        'VE-2026-23255,f613e8b4afea0cd17c7168e8b00e25bc8d33175d',
        '',
        'CVE-2025-39808,185c926283da67a72df20a63a5046b3b4631b7d9',
        'CVE-2025-39808,185c926283da67a72df20a63a5046b3b4631b7d9',
        'CVE-2025-39891,0e20450829ca3c1dbc2db536391537c57a40fe0b',
    ]

    cases = iter_patch_cases(lines)

    assert len(cases) == 2
    assert cases[0].cve_id == 'CVE-2025-39808'
    assert cases[0].fix_commit == '185c926283da67a72df20a63a5046b3b4631b7d9'
    assert cases[1].cve_id == 'CVE-2025-39891'
