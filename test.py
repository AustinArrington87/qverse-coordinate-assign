import httpx
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

async def test_coordinate_assignment(data: Dict[str, Any]):
    url = "http://127.0.0.1:8001/assign-coordinates"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nSuccess for {data.get('category')}/{data.get('subcategory')}/{data.get('sub_subcategory')}:")
                print(f"Coordinates: x={result['x']:.2f}, y={result['y']:.2f}, z={result['z']:.2f}")
                print(f"Timestamp: {result['timestamp']}")
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

async def test_get_zones():
    url = "http://127.0.0.1:8001/zones"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                zones = response.json()
                print("\nAvailable zones:")
                for category, details in zones.items():
                    print(f"\nCategory: {category}")
                    if "subcategories" in details:
                        for subcategory, subdetails in details["subcategories"].items():
                            print(f"  Subcategory: {subcategory}")
                            if "sub_subcategories" in subdetails:
                                for sub_sub in subdetails["sub_subcategories"].keys():
                                    print(f"    Sub-subcategory: {sub_sub}")
            else:
                print(f"Error getting zones: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

async def test_get_history():
    url = "http://127.0.0.1:8001/coordinate-history"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                history = response.json()
                print("\nCoordinate History:")
                for record in history:
                    print("\nRecord:")
                    print(f"Category: {record['input_data']['category']}")
                    print(f"Coordinates: {record['coordinates']}")
                    print(f"Timestamp: {record['temporal']['datetime']}")
            else:
                print(f"Error getting history: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

async def test_filtered_history():
    # Get today's date
    today = datetime.now().date()
    
    # Test date range filtering
    params = {
        'start_date': (today - timedelta(days=7)).isoformat(),
        'end_date': today.isoformat(),
        'category': 'science_nature'
    }
    
    url = "http://127.0.0.1:8001/coordinate-history"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                history = response.json()
                print(f"\nFiltered History (Last 7 days, Science Nature):")
                print(f"Found {len(history)} records")
                for record in history[:3]:  # Show first 3 records
                    print("\nRecord:")
                    print(f"Category: {record['input_data']['category']}")
                    print(f"Coordinates: {record['coordinates']}")
                    print(f"Timestamp: {record['temporal']['datetime']}")
            else:
                print(f"Error getting filtered history: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

async def clear_history():
    url = "http://127.0.0.1:8001/coordinate-history"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url)
            if response.status_code == 200:
                print("\nHistory cleared successfully")
            else:
                print(f"Error clearing history: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

async def run_tests():
    print("Starting Q-verse coordinate assignment tests...")
    
    # Optionally clear history before running tests
    # await clear_history()
    
    # Test getting all zones first
    await test_get_zones()
    
    # Test cases for different categories
    test_cases = [
        # Science test cases
        {
            "vector": [0.5, 0.5, 0.5],
            "category": "science_nature",
            "subcategory": "physics",
            "sub_subcategory": "quantum",
            "metadata": {
                "description": "Quantum physics experiment data",
                "researcher": "Dr. Smith",
                "lab_id": "QP001"
            }
        },
        {
            "vector": [0.7, 0.3, 0.6],
            "category": "science_nature",
            "subcategory": "astronomy",
            "sub_subcategory": "cosmology",
            "metadata": {
                "telescope": "Hubble",
                "observation_id": "COSMOS123"
            }
        },
        
        # Sports test cases
        {
            "vector": [0.2, 0.8, 0.4],
            "category": "sports_recreation",
            "subcategory": "team_sports",
            "sub_subcategory": "basketball",
            "detail": "nba",
            "metadata": {
                "game_id": "NBA20241112",
                "team": "Lakers"
            }
        },
        {
            "vector": [0.9, 0.1, 0.5],
            "category": "sports_recreation",
            "subcategory": "combat_sports",
            "sub_subcategory": "boxing",
            "detail": "professional",
            "metadata": {
                "event_id": "BOX20241112",
                "weight_class": "heavyweight"
            }
        },
        
        # Test with just category and subcategory
        {
            "vector": [0.4, 0.6, 0.3],
            "category": "arts_entertainment",
            "subcategory": "visual_arts",
            "metadata": {
                "art_type": "painting",
                "medium": "oil"
            }
        }
    ]
    
    # Run all test cases
    for test_case in test_cases:
        await test_coordinate_assignment(test_case)
    
    # Test getting history
    print("\nTesting history retrieval...")
    await test_get_history()
    
    # Test filtered history
    print("\nTesting filtered history retrieval...")
    await test_filtered_history()

if __name__ == "__main__":
    asyncio.run(run_tests())