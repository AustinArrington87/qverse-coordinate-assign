from typing import Dict, Tuple, Optional, Any

# System Constants
COORDINATE_SYSTEM = {
    "min_bound": 501,
    "max_bound": 1501,
    "center": (1001, 1001, 1001),
    "decimal_places": 12,
    "total_span": 1000  # 1501 - 501 = 1000km span
}

# Helper function to calculate ranges
def calculate_range(start: int, span: int) -> Tuple[int, int]:
    """Calculate a range given a start point and span."""
    return (start, start + span)

zones: Dict[str, Any] = {
    "arts_entertainment": {
        "x_range": (501, 601),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "visual_arts": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "painting": (501, 511),
                    "sculpture": (512, 522),
                    "photography": (523, 534)
                }
            },
            "music": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "classical": (535, 545),
                    "popular": (546, 556),
                    "jazz": (557, 567)
                }
            },
            "film_tv": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "movies": (568, 578),
                    "television": (579, 589),
                    "streaming": (590, 601)
                }
            }
        }
    },
    "business_finance": {
        "x_range": (602, 702),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "ecommerce": {
                "z_subrange": (501, 521),
                "sub_subcategories": {
                    "retail": {
                        "range": (501, 507),
                        "details": {"b2c": (501, 504), "b2b": (505, 507)}
                    },
                    "marketplace": (508, 514),
                    "digital_goods": (515, 521)
                }
            },
            "banking": {
                "z_subrange": (522, 541),
                "sub_subcategories": {
                    "commercial": (522, 528),
                    "investment": (529, 535),
                    "retail": (536, 541)
                }
            },
            "insurance": {
                "z_subrange": (542, 561),
                "sub_subcategories": {
                    "life": (542, 548),
                    "health": (549, 555),
                    "property": (556, 561)
                }
            },
            "real_estate": {
                "z_subrange": (562, 581),
                "sub_subcategories": {
                    "residential": (562, 568),
                    "commercial": (569, 575),
                    "industrial": (576, 581)
                }
            },
            "investment": {
                "z_subrange": (582, 601),
                "sub_subcategories": {
                    "stocks": (582, 588),
                    "bonds": (589, 595),
                    "crypto": (596, 601)
                }
            }
        }
    },
    "communications": {
        "x_range": (703, 803),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "telecom": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "mobile": (501, 511),
                    "internet": (512, 522),
                    "infrastructure": (523, 534)
                }
            },
            "marketing": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "digital": (535, 545),
                    "traditional": (546, 556),
                    "social_media": (557, 567)
                }
            },
            "media": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "news": (568, 578),
                    "publishing": (579, 589),
                    "digital_content": (590, 601)
                }
            }
        }
    },
    "construction": {
        "x_range": (804, 904),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "residential": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "houses": (501, 511),
                    "apartments": (512, 522),
                    "renovations": (523, 534)
                }
            },
            "commercial": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "offices": (535, 545),
                    "retail_spaces": (546, 556),
                    "industrial": (557, 567)
                }
            },
            "infrastructure": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "transportation": (568, 578),
                    "utilities": (579, 589),
                    "public_works": (590, 601)
                }
            }
        }
    },
    "energy": {
        "x_range": (905, 1005),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "renewable": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "solar": (501, 511),
                    "wind": (512, 522),
                    "hydro": (523, 534)
                }
            },
            "fossil_fuels": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "oil": (535, 545),
                    "gas": (546, 556),
                    "coal": (557, 567)
                }
            },
            "nuclear": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "fission": (568, 584),
                    "research": (585, 601)
                }
            }
        }
    },
    "fashion_retail": {
        "x_range": (1006, 1106),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "apparel": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "luxury": (501, 511),
                    "casual": (512, 522),
                    "sportswear": (523, 534)
                }
            },
            "accessories": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "jewelry": (535, 545),
                    "bags": (546, 556),
                    "watches": (557, 567)
                }
            },
            "footwear": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "casual": (568, 578),
                    "athletic": (579, 589),
                    "formal": (590, 601)
                }
            }
        }
    },
    "food_beverage": {
        "x_range": (1107, 1207),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "restaurants": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "fine_dining": (501, 511),
                    "fast_food": (512, 522),
                    "casual_dining": (523, 534)
                }
            },
            "agriculture": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "farming": (535, 545),
                    "livestock": (546, 556),
                    "sustainable": (557, 567)
                }
            },
            "processing": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "packaged_foods": (568, 578),
                    "beverages": (579, 589),
                    "dairy": (590, 601)
                }
            }
        }
    },
    "health_wellness": {
        "x_range": (1208, 1308),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "medical": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "hospitals": (501, 511),
                    "clinics": (512, 522),
                    "research": (523, 534)
                }
            },
            "fitness": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "gyms": (535, 545),
                    "equipment": (546, 556),
                    "training": (557, 567)
                }
            },
            "mental_health": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "therapy": (568, 578),
                    "counseling": (579, 589),
                    "wellness": (590, 601)
                }
            }
        }
    },
    "manufacturing": {
        "x_range": (1309, 1409),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "automotive": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "vehicles": (501, 511),
                    "parts": (512, 522),
                    "equipment": (523, 534)
                }
            },
            "electronics": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "consumer": (535, 545),
                    "industrial": (546, 556),
                    "components": (557, 567)
                }
            },
            "machinery": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "industrial": (568, 578),
                    "agricultural": (579, 589),
                    "construction": (590, 601)
                }
            }
        }
    },
    "technology": {
        "x_range": (1410, 1501),
        "y_range": (501, 601),
        "z_range": (501, 601),
        "subcategories": {
            "software": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "enterprise": (501, 511),
                    "consumer": (512, 522),
                    "mobile": (523, 534)
                }
            },
            "hardware": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "computers": (535, 545),
                    "mobile": (546, 556),
                    "networking": (557, 567)
                }
            },
            "ai_ml": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "applications": (568, 578),
                    "research": (579, 589),
                    "infrastructure": (590, 601)
                }
            }
        }
    },
    "transportation": {
        "x_range": (501, 601),
        "y_range": (602, 702),
        "z_range": (501, 601),
        "subcategories": {
            "automotive": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "personal": (501, 511),
                    "commercial": (512, 522),
                    "services": (523, 534)
                }
            },
            "aviation": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "commercial": (535, 545),
                    "private": (546, 556),
                    "cargo": (557, 567)
                }
            },
            "shipping": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "maritime": (568, 578),
                    "logistics": (579, 589),
                    "delivery": (590, 601)
                }
            }
        }
    },
    "tourism_hospitality": {
        "x_range": (602, 702),
        "y_range": (602, 702),
        "z_range": (501, 601),
        "subcategories": {
            "hotels": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "luxury": (501, 511),
                    "business": (512, 522),
                    "budget": (523, 534)
                }
            },
            "travel": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "air": (535, 545),
                    "cruise": (546, 556),
                    "packages": (557, 567)
                }
            },
            "attractions": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "theme_parks": (568, 578),
                    "museums": (579, 589),
                    "cultural": (590, 601)
                }
            }
        }
    },
    "sports_recreation": {
        "x_range": (703, 803),
        "y_range": (602, 702),
        "z_range": (501, 601),
        "subcategories": {
            "team_sports": {
                "z_subrange": (501, 534),
                "sub_subcategories": {
                    "basketball": {
                        "range": (501, 511),
                        "details": {
                            "nba": (501, 506),
                            "international": (507, 511)
                        }
                    },
                    "football": {
                        "range": (512, 522),
                        "details": {
                            "nfl": (512, 517),
                            "soccer": (518, 522)
                        }
                    },
                    "baseball": {
                        "range": (523, 534),
                        "details": {
                            "mlb": (523, 528),
                            "international": (529, 534)
                        }
                    }
                }
            },
            "combat_sports": {
                "z_subrange": (535, 567),
                "sub_subcategories": {
                    "boxing": {
                        "range": (535, 545),
                        "details": {
                            "professional": (535, 540),
                            "amateur": (541, 545)
                        }
                    },
                    "mma": {
                        "range": (546, 556),
                        "details": {
                            "ufc": (546, 551),
                            "other": (552, 556)
                        }
                    },
                    "wrestling": {
                        "range": (557, 567),
                        "details": {
                            "freestyle": (557, 562),
                            "greco_roman": (563, 567)
                        }
                    }
                }
            },
            "individual_sports": {
                "z_subrange": (568, 601),
                "sub_subcategories": {
                    "tennis": (568, 578),
                    "golf": (579, 589),
                    "athletics": (590, 601)
                }
            }
        }
    },
    "science_nature": {
        "x_range": (804, 904),
        "y_range": (602, 702),
        "z_range": (501, 601),
        "subcategories": {
            "physics": {
                "z_subrange": (501, 517),
                "sub_subcategories": {
                    "quantum": (501, 506),
                    "mechanics": (507, 511),
                    "relativity": (512, 517)
                }
            },
            "chemistry": {
                "z_subrange": (518, 534),
                "sub_subcategories": {
                    "organic": (518, 523),
                    "inorganic": (524, 529),
                    "physical": (530, 534)
                }
            },
            "biology": {
                "z_subrange": (535, 551),
                "sub_subcategories": {
                    "molecular": (535, 540),
                    "genetics": (541, 546),
                    "ecology": (547, 551)
                }
            },
            "environmental": {
                "z_subrange": (552, 567),
                "sub_subcategories": {
                    "climate": (552, 557),
                    "conservation": (558, 562),
                    "sustainability": (563, 567)
                }
            },
            "astronomy": {
                "z_subrange": (568, 584),
                "sub_subcategories": {
                    "cosmology": (568, 573),
                    "planetary": (574, 579),
                    "astrophysics": (580, 584)
                }
            },
            "geology": {
                "z_subrange": (585, 601),
                "sub_subcategories": {
                    "mineralogy": (585, 590),
                    "tectonics": (591, 596),
                    "paleontology": (597, 601)
                }
            }
        }
    }
}

# Helper Functions
def validate_coordinate(coord: float) -> float:
    """Ensure coordinate is within valid bounds and properly rounded."""
    validated = max(COORDINATE_SYSTEM["min_bound"], 
                   min(COORDINATE_SYSTEM["max_bound"], coord))
    return round(validated, COORDINATE_SYSTEM["decimal_places"])

def get_subcategory_range(
    category_range: Tuple[int, int],
    subcategory_index: int,
    total_subcategories: int
) -> Tuple[int, int]:
    """Calculate range for subcategory within its parent category range."""
    span = category_range[1] - category_range[0]
    segment = span / total_subcategories
    start = category_range[0] + (subcategory_index * segment)
    end = start + segment
    return (int(start), int(end))

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

def calculate_center_point(range_tuple: Tuple[int, int]) -> float:
    """Calculate the center point of a range."""
    return validate_coordinate((range_tuple[0] + range_tuple[1]) / 2)

def get_category_center(category: str) -> Tuple[float, float, float]:
    """Get the center point coordinates for a category."""
    if not validate_category(category):
        raise ValueError(f"Invalid category: {category}")
        
    zone = zones[category]
    return (
        calculate_center_point(zone["x_range"]),
        calculate_center_point(zone["y_range"]),
        calculate_center_point(zone["z_range"])
    )

def verify_zones() -> bool:
    """Verify zones configuration."""
    try:
        assert isinstance(zones, dict), "zones must be a dictionary"
        assert len(zones) > 0, "zones dictionary is empty"
        for category, config in zones.items():
            assert "x_range" in config, f"x_range missing for {category}"
            assert "y_range" in config, f"y_range missing for {category}"
            assert "z_range" in config, f"z_range missing for {category}"
            assert "subcategories" in config, f"subcategories missing for {category}"
        print(f"Zones verification successful. Found {len(zones)} categories:")
        for category in zones:
            print(f"  - {category}")
        return True
    except Exception as e:
        print(f"Zones verification failed: {e}")
        return False

# Export necessary components
__all__ = [
    'zones',
    'COORDINATE_SYSTEM',
    'validate_category',
    'validate_subcategory',
    'get_coordinate_range',
    'validate_coordinate',
    'get_category_center',
    'verify_zones'
]

if __name__ == "__main__":
    verify_zones()
