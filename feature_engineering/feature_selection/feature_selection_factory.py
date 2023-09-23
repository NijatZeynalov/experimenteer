from feature_engineering.feature_selection.feature_selectors import (
    KBestSelector,
    RecursiveFeatureSelector,
    DummySelector,
)


SELECTORS = {
    "select_k_best": KBestSelector,
    "recursive_feature_elimination": RecursiveFeatureSelector,
    "none": DummySelector,
}


class FeatureSelectionFactory:
    @staticmethod
    def get_selector(selector_type):
        if selector_type not in SELECTORS:
            raise ValueError(
                f"Invalid Selector type: {selector_type}. "
                f"Supported selectors: {list(SELECTORS.keys())}"
            )

        return SELECTORS[selector_type]()
