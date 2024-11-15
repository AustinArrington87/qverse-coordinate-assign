from typing import Dict, Tuple, Optional, Any

# Define zones configuration
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

def validate_category(category: str) -> bool:
    """Validate if category exists in zones."""
    return category in zones

def validate_subcategory(category: str, subcategory: str) -> bool:
    """Validate if subcategory exists in given category."""
    return category in zones and subcategory in zones[category]["subcategories"]

def get_coordinate_range(
    category: str, 
    subcategory: Optional[str] = None,
    sub_subcategory: Optional[str] = None,
    detail: Optional[str] = None
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Get coordinate ranges for the specified category hierarchy."""
    if not validate_category(category):
        raise ValueError(f"Invalid category: {category}")
    
    zone = zones[category]
    x_range = zone["x_range"]
    y_range = zone["y_range"]
    
    if not subcategory:
        return x_range, y_range, zone["z_range"]
    
    if not validate_subcategory(category, subcategory):
        raise ValueError(f"Invalid subcategory: {subcategory}")
    
    subcat = zone["subcategories"][subcategory]
    
    if not sub_subcategory:
        return x_range, y_range, subcat["z_subrange"]
    
    if sub_subcategory not in subcat["sub_subcategories"]:
        raise ValueError(f"Invalid sub-subcategory: {sub_subcategory}")
    
    sub_subcat = subcat["sub_subcategories"][sub_subcategory]
    
    if isinstance(sub_subcat, tuple):
        return x_range, y_range, sub_subcat
    elif isinstance(sub_subcat, dict):
        if not detail:
            return x_range, y_range, sub_subcat["range"]
        if detail not in sub_subcat["details"]:
            raise ValueError(f"Invalid detail: {detail}")
        return x_range, y_range, sub_subcat["details"][detail]
    
    raise ValueError("Invalid zone configuration")

# Export necessary components
__all__ = ['zones', 'validate_category', 'validate_subcategory', 'get_coordinate_range']
