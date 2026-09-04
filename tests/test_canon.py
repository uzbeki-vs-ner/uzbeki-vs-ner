# ruff: noqa: RUF001
"""CPU tests for v0 entity canon (suffix strip + apostrophe fold)."""

from uzbek_ner.canon import canon_surface, mention_key


def test_toshkentda_to_toshkent() -> None:
    assert canon_surface("Toshkentda") == "Toshkent"


def test_qodirovning_to_qodirov() -> None:
    assert canon_surface("Qodirovning") == "Qodirov"


def test_kfcda_to_kfc() -> None:
    assert canon_surface("KFCda") == "KFC"


def test_spaced_kfc_da_untouched() -> None:
    assert canon_surface("KFC da") == "KFC da"


def test_viloyati_not_stripped() -> None:
    assert canon_surface("Farg'ona viloyati") == "Farg'ona viloyati"


def test_viloyatida_keeps_admin_word() -> None:
    assert canon_surface("Farg'ona viloyatida") == "Farg'ona viloyati"


def test_andijon_geo_org_keys_differ() -> None:
    surface = "Andijon"
    assert mention_key("GEO", surface) != mention_key("ORG", surface)
    assert mention_key("GEO", surface) == ("GEO", "andijon")
    assert mention_key("ORG", "Andijonda") == ("ORG", "andijon")


def test_apostrophe_fold_and_casefold_key() -> None:
    assert canon_surface("Oʻzbekistonda") == "O'zbekiston"
    assert mention_key("GEO", "Oʻzbekistonda") == mention_key("GEO", "O'zbekiston")
    assert mention_key("GEO", "Toshkentda") == mention_key("GEO", "toshkent")


def test_cyrillic_locative() -> None:
    assert canon_surface("Тошкентда") == "Тошкент"


def test_kanada_lemma_not_stripped() -> None:
    assert canon_surface("Kanada") == "Kanada"
    assert canon_surface("Kanadada") == "Kanada"
