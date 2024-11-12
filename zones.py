from typing import Dict, Any

zones: Dict[str, Any] = {
    "arts_entertainment": {
        "x_range": (0, 50), 
        "y_range": (0, 50), 
        "z_range": (0, 50),
        "subcategories": {
            "visual_arts": {"z_subrange": (0, 16)},
            "music": {"z_subrange": (17, 33)},
            "film_tv": {"z_subrange": (34, 50)}
        }
    },
    "science_nature": {
        "x_range": (0, 50), 
        "y_range": (101, 150), 
        "z_range": (0, 50),
        "subcategories": {
            "physics": {
                "z_subrange": (0, 8),
                "sub_subcategories": {
                    "quantum": (0, 2),
                    "mechanics": (3, 5),
                    "relativity": (6, 8)
                }
            },
            "chemistry": {
                "z_subrange": (9, 16),
                "sub_subcategories": {
                    "organic": (9, 11),
                    "inorganic": (12, 14),
                    "physical": (15, 16)
                }
            },
            "biology": {
                "z_subrange": (17, 24),
                "sub_subcategories": {
                    "molecular": (17, 19),
                    "genetics": (20, 22),
                    "ecology": (23, 24)
                }
            },
            "astronomy": {
                "z_subrange": (25, 32),
                "sub_subcategories": {
                    "cosmology": (25, 27),
                    "planetary": (28, 30),
                    "astrophysics": (31, 32)
                }
            },
            "geology": {
                "z_subrange": (33, 40),
                "sub_subcategories": {
                    "mineralogy": (33, 35),
                    "tectonics": (36, 38),
                    "paleontology": (39, 40)
                }
            },
            "environmental": {
                "z_subrange": (41, 50),
                "sub_subcategories": {
                    "climate": (41, 44),
                    "conservation": (45, 47),
                    "sustainability": (48, 50)
                }
            }
        }
    },
    "sports_recreation": {
        "x_range": (151, 200), 
        "y_range": (51, 100), 
        "z_range": (0, 50),
        "subcategories": {
            "team_sports": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "basketball": {
                        "range": (0, 3),
                        "details": {
                            "nba": (0, 1),
                            "international": (2, 3)
                        }
                    },
                    "football": {
                        "range": (4, 7),
                        "details": {
                            "nfl": (4, 5),
                            "soccer": (6, 7)
                        }
                    },
                    "baseball": {
                        "range": (8, 11),
                        "details": {
                            "mlb": (8, 9),
                            "international": (10, 11)
                        }
                    },
                    "hockey": {
                        "range": (12, 16),
                        "details": {
                            "nhl": (12, 14),
                            "international": (15, 16)
                        }
                    }
                }
            },
            "combat_sports": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "boxing": {
                        "range": (17, 21),
                        "details": {
                            "professional": (17, 19),
                            "amateur": (20, 21)
                        }
                    },
                    "mma": {
                        "range": (22, 26),
                        "details": {
                            "ufc": (22, 24),
                            "other": (25, 26)
                        }
                    },
                    "wrestling": {
                        "range": (27, 30),
                        "details": {
                            "freestyle": (27, 28),
                            "greco_roman": (29, 30)
                        }
                    },
                    "martial_arts": {
                        "range": (31, 33),
                        "details": {
                            "karate": (31, 32),
                            "judo": (32, 33)
                        }
                    }
                }
            },
            "individual_sports": {
                "z_subrange": (34, 42),
                "sub_subcategories": {
                    "tennis": (34, 36),
                    "golf": (37, 39),
                    "athletics": (40, 42)
                }
            },
            "outdoor_activities": {
                "z_subrange": (43, 50),
                "sub_subcategories": {
                    "hiking": (43, 44),
                    "climbing": (45, 46),
                    "cycling": (47, 48),
                    "water_sports": (49, 50)
                }
            }
        }
    }
}

# Helper functions for zone operations
def validate_category(category: str) -> bool:
    return category in zones

def validate_subcategory(category: str, subcategory: str) -> bool:
    return category in zones and subcategory in zones[category]["subcategories"]

def get_zone_ranges(category: str):
    if not validate_category(category):
        raise ValueError(f"Invalid category: {category}")
    return zones[category]["x_range"], zones[category]["y_range"], zones[category]["z_range"]