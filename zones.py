from typing import Dict, Tuple, Optional, Any

zones: Dict[str, Any] = {
    "arts_entertainment": {
        "x_range": (0, 50),
        "y_range": (0, 50),
        "z_range": (0, 50),
        "subcategories": {
            "visual_arts": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "painting": (0, 5),
                    "sculpture": (6, 10),
                    "photography": (11, 16)
                }
            },
            "music": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "classical": (17, 22),
                    "popular": (23, 28),
                    "jazz": (29, 33)
                }
            },
            "film_tv": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "movies": (34, 39),
                    "television": (40, 45),
                    "streaming": (46, 50)
                }
            }
        }
    },
    "business_finance": {
        "x_range": (51, 100),
        "y_range": (0, 50),
        "z_range": (0, 50),
        "subcategories": {
            "ecommerce": {
                "z_subrange": (0, 12),
                "sub_subcategories": {
                    "retail": {
                        "range": (0, 4),
                        "details": {"b2c": (0, 4)}
                    },
                    "wholesale": {
                        "range": (5, 8),
                        "details": {"b2b": (5, 8)}
                    },
                    "marketplace": (9, 12)
                }
            },
            "banking": {
                "z_subrange": (13, 25),
                "sub_subcategories": {
                    "commercial": (13, 16),
                    "investment": (17, 21),
                    "retail": (22, 25)
                }
            },
            "investment": {
                "z_subrange": (26, 33),
                "sub_subcategories": {
                    "stocks": (26, 28),
                    "bonds": (29, 31),
                    "crypto": (32, 33)
                }
            },
            "insurance": {
                "z_subrange": (34, 42),
                "sub_subcategories": {
                    "life": (34, 36),
                    "health": (37, 39),
                    "property": (40, 42)
                }
            },
            "real_estate": {
                "z_subrange": (43, 50),
                "sub_subcategories": {
                    "residential": (43, 45),
                    "commercial": (46, 48),
                    "industrial": (49, 50)
                }
            }
        }
    },
    "communications": {
        "x_range": (101, 150),
        "y_range": (0, 50),
        "z_range": (0, 50),
        "subcategories": {
            "telecom": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "mobile": (0, 5),
                    "internet": (6, 11),
                    "infrastructure": (12, 16)
                }
            },
            "marketing": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "digital": (17, 22),
                    "traditional": (23, 28),
                    "social_media": (29, 33)
                }
            }
        }
    },
    "construction": {
        "x_range": (151, 200),
        "y_range": (0, 50),
        "z_range": (0, 50),
        "subcategories": {
            "residential": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "houses": (0, 8),
                    "apartments": (9, 16)
                }
            },
            "commercial": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "offices": (17, 25),
                    "retail": (26, 33)
                }
            },
            "industrial": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "factories": (34, 42),
                    "warehouses": (43, 50)
                }
            }
        }
    },
    "energy": {
        "x_range": (0, 50),
        "y_range": (51, 100),
        "z_range": (0, 50),
        "subcategories": {
            "renewable": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "solar": (0, 5),
                    "wind": (6, 11),
                    "hydro": (12, 16)
                }
            },
            "fossil_fuels": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "oil": (17, 22),
                    "gas": (23, 28),
                    "coal": (29, 33)
                }
            },
            "nuclear": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "fission": (34, 42),
                    "research": (43, 50)
                }
            }
        }
    },
    "fashion_retail": {
        "x_range": (51, 100),
        "y_range": (51, 100),
        "z_range": (0, 50),
        "subcategories": {
            "apparel": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "luxury": (0, 8),
                    "casual": (9, 16)
                }
            },
            "accessories": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "jewelry": (17, 25),
                    "bags": (26, 33)
                }
            },
            "footwear": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "casual": (34, 42),
                    "athletic": (43, 50)
                }
            }
        }
    },
    "food_beverage": {
        "x_range": (101, 150),
        "y_range": (51, 100),
        "z_range": (0, 50),
        "subcategories": {
            "restaurants": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "fine_dining": (0, 5),
                    "fast_food": (6, 11),
                    "casual_dining": (12, 16)
                }
            },
            "agriculture": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "farming": (17, 25),
                    "livestock": (26, 33)
                }
            },
            "processing": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "packaged_foods": (34, 42),
                    "beverages": (43, 50)
                }
            }
        }
    },
    "health_wellness": {
        "x_range": (151, 200),
        "y_range": (51, 100),
        "z_range": (0, 50),
        "subcategories": {
            "medical": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "hospitals": (0, 5),
                    "clinics": (6, 11),
                    "research": (12, 16)
                }
            },
            "fitness": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "gyms": (17, 25),
                    "equipment": (26, 33)
                }
            },
            "mental_health": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "therapy": (34, 42),
                    "counseling": (43, 50)
                }
            }
        }
    },
    "manufacturing": {
        "x_range": (0, 50),
        "y_range": (101, 150),
        "z_range": (0, 50),
        "subcategories": {
            "automotive": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "vehicles": (0, 8),
                    "parts": (9, 16)
                }
            },
            "electronics": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "consumer": (17, 25),
                    "industrial": (26, 33)
                }
            },
            "machinery": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "industrial": (34, 42),
                    "agricultural": (43, 50)
                }
            }
        }
    },
    "science_nature": {
        "x_range": (51, 100),
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
                "z_subrange": (17, 25),
                "sub_subcategories": {
                    "molecular": (17, 19),
                    "genetics": (20, 22),
                    "ecology": (23, 25)
                }
            },
            "astronomy": {
                "z_subrange": (26, 33),
                "sub_subcategories": {
                    "cosmology": (26, 28),
                    "planetary": (29, 31),
                    "astrophysics": (32, 33)
                }
            },
            "geology": {
                "z_subrange": (34, 42),
                "sub_subcategories": {
                    "mineralogy": (34, 36),
                    "tectonics": (37, 39),
                    "paleontology": (40, 42)
                }
            },
            "environmental": {
                "z_subrange": (43, 50),
                "sub_subcategories": {
                    "climate": (43, 45),
                    "conservation": (46, 48),
                    "sustainability": (49, 50)
                }
            }
        }
    },
    "sports_recreation": {
        "x_range": (101, 150),
        "y_range": (101, 150),
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
            }
        }
    },
    "technology": {
        "x_range": (151, 200),
        "y_range": (101, 150),
        "z_range": (0, 50),
        "subcategories": {
            "software": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "enterprise": (0, 5),
                    "consumer": (6, 11),
                    "mobile": (12, 16)
                }
            },
            "hardware": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "computers": (17, 22),
                    "mobile": (23, 28),
                    "networking": (29, 33)
                }
            },
            "ai_ml": {
                "z_subrange": (34, 42),
                "sub_subcategories": {
                    "applications": (34, 38),
                    "research": (39, 42)
                }
            },
            "cloud": {
                "z_subrange": (43, 50),
                "sub_subcategories": {
                    "infrastructure": (43, 46),
                    "services": (47, 50)
                }
            }
        }
    },
    "tourism_hospitality": {
        "x_range": (0, 50),
        "y_range": (151, 200),
        "z_range": (0, 50),
        "subcategories": {
            "hotels": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "luxury": (0, 8),
                    "budget": (9, 16)
                }
            },
            "travel": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "air": (17, 25),
                    "cruise": (26, 33)
                }
            },
            "attractions": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "theme_parks": (34, 42),
                    "museums": (43, 50)
                }
            }
        }
    },
    "transportation": {
        "x_range": (51, 100),
        "y_range": (151, 200),
        "z_range": (0, 50),
        "subcategories": {
            "automotive": {
                "z_subrange": (0, 16),
                "sub_subcategories": {
                    "personal": (0, 8),
                    "commercial": (9, 16)
                }
            },
            "aviation": {
                "z_subrange": (17, 33),
                "sub_subcategories": {
                    "commercial": (17, 25),
                    "private": (26, 33)
                }
            },
            "shipping": {
                "z_subrange": (34, 50),
                "sub_subcategories": {
                    "maritime": (34, 42),
                    "logistics": (43, 50)
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
