#!/usr/bin/env python3
"""
Dynamic Pricing API - Simple REST-like interface for pricing operations
This module provides a simple API-style interface for the pricing system.
"""

from typing import Dict, Any, Optional
from pricing_system import rule_based_pricing, predict_optimal_price
import json
from datetime import datetime


class PricingAPI:
    """
    Simple API class for dynamic pricing operations
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.system_name = "Elves' Dynamic Pricing System"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "available_endpoints": [
                "/price/rule-based",
                "/price/ai-optimized", 
                "/price/compare-strategies",
                "/system/info"
            ]
        }
    
    def calculate_rule_based_price(self, 
                                  original_price: float,
                                  inventory_level: int,
                                  is_holiday: bool = False,
                                  is_weekend: bool = False,
                                  customer_segment: str = "Regular") -> Dict[str, Any]:
        """
        Calculate price using rule-based algorithm
        
        Returns API-formatted response
        """
        try:
            price, adjustments = rule_based_pricing(
                original_price, inventory_level, is_holiday, is_weekend, customer_segment
            )
            
            change_percent = ((price / original_price) - 1) * 100
            
            return {
                "success": True,
                "method": "rule-based",
                "input": {
                    "original_price": original_price,
                    "inventory_level": inventory_level,
                    "is_holiday": is_holiday,
                    "is_weekend": is_weekend,
                    "customer_segment": customer_segment
                },
                "result": {
                    "recommended_price": price,
                    "price_change_percent": round(change_percent, 2),
                    "applied_adjustments": adjustments,
                    "confidence": 1.0  # Rule-based has full confidence
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "rule-based",
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_ai_optimized_price(self,
                                   original_price: float,
                                   inventory_level: int,
                                   is_holiday: bool = False,
                                   is_weekend: bool = False,
                                   competitor_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate price using AI optimization
        
        Returns API-formatted response
        """
        try:
            # Use original price as competitor price if not provided
            if competitor_price is None:
                competitor_price = original_price * 1.05  # Assume 5% higher competitor price
            
            result = predict_optimal_price(
                original_price, inventory_level, is_holiday, is_weekend, competitor_price
            )
            
            change_percent = ((result['predicted_price'] / original_price) - 1) * 100
            
            return {
                "success": True,
                "method": "ai-optimized",
                "input": {
                    "original_price": original_price,
                    "inventory_level": inventory_level,
                    "is_holiday": is_holiday,
                    "is_weekend": is_weekend,
                    "competitor_price": competitor_price
                },
                "result": {
                    "recommended_price": result['predicted_price'],
                    "price_change_percent": round(change_percent, 2),
                    "confidence": result['confidence_score'],
                    "expected_revenue": result['revenue_estimate'],
                    "demand_forecast": result['demand_forecast'],
                    "price_sensitivity": result['price_sensitivity']
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "ai-optimized",
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_pricing_methods(self,
                               original_price: float,
                               inventory_level: int,
                               is_holiday: bool = False,
                               is_weekend: bool = False,
                               customer_segment: str = "Regular",
                               competitor_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare different pricing methods for the same inputs
        
        Returns comparison of all methods
        """
        try:
            # Get rule-based result
            rule_result = self.calculate_rule_based_price(
                original_price, inventory_level, is_holiday, is_weekend, customer_segment
            )
            
            # Get AI result
            ai_result = self.calculate_ai_optimized_price(
                original_price, inventory_level, is_holiday, is_weekend, competitor_price
            )
            
            # Calculate static pricing result
            static_result = {
                "recommended_price": original_price,
                "price_change_percent": 0.0,
                "confidence": 1.0,
                "applied_adjustments": []
            }
            
            # Determine recommendation
            if ai_result["success"] and rule_result["success"]:
                ai_price = ai_result["result"]["recommended_price"]
                rule_price = rule_result["result"]["recommended_price"]
                
                # Recommend method with higher expected revenue
                if "expected_revenue" in ai_result["result"]:
                    recommended_method = "ai-optimized"
                    recommended_price = ai_price
                elif abs(rule_price - original_price) > abs(ai_price - original_price):
                    recommended_method = "rule-based"
                    recommended_price = rule_price
                else:
                    recommended_method = "ai-optimized"
                    recommended_price = ai_price
            else:
                recommended_method = "static"
                recommended_price = original_price
            
            return {
                "success": True,
                "method": "comparison",
                "input": {
                    "original_price": original_price,
                    "inventory_level": inventory_level,
                    "is_holiday": is_holiday,
                    "is_weekend": is_weekend,
                    "customer_segment": customer_segment,
                    "competitor_price": competitor_price
                },
                "results": {
                    "static_pricing": static_result,
                    "rule_based": rule_result["result"] if rule_result["success"] else {"error": rule_result.get("error")},
                    "ai_optimized": ai_result["result"] if ai_result["success"] else {"error": ai_result.get("error")}
                },
                "recommendation": {
                    "method": recommended_method,
                    "price": recommended_price
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "comparison",
                "timestamp": datetime.now().isoformat()
            }


def demo_api():
    """Demonstrate the API functionality"""
    print("🚀 Dynamic Pricing API Demo")
    print("=" * 40)
    
    api = PricingAPI()
    
    # System info
    print("\n📋 System Information:")
    info = api.get_system_info()
    print(json.dumps(info, indent=2))
    
    # Test cases
    test_cases = [
        {
            "name": "Holiday Weekend Rush",
            "params": {
                "original_price": 75.0,
                "inventory_level": 8,
                "is_holiday": True,
                "is_weekend": True,
                "customer_segment": "Regular",
                "competitor_price": 78.0
            }
        },
        {
            "name": "Clearance Sale",
            "params": {
                "original_price": 50.0,
                "inventory_level": 85,
                "is_holiday": False,
                "is_weekend": False,
                "customer_segment": "Loyal",
                "competitor_price": 45.0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Test Case: {test_case['name']}")
        print("-" * 30)
        
        # Compare all methods
        result = api.compare_pricing_methods(**test_case["params"])
        
        if result["success"]:
            print(f"💡 Recommendation: {result['recommendation']['method']} → ${result['recommendation']['price']:.2f}")
            
            print("\n📊 Method Comparison:")
            results = result["results"]
            
            # Static pricing
            static = results["static_pricing"]
            print(f"  📏 Static: ${static['recommended_price']:.2f} ({static['price_change_percent']:+.1f}%)")
            
            # Rule-based
            if "error" not in results["rule_based"]:
                rule = results["rule_based"]
                print(f"  📚 Rule-based: ${rule['recommended_price']:.2f} ({rule['price_change_percent']:+.1f}%)")
                if rule.get("applied_adjustments"):
                    print(f"       Adjustments: {', '.join(rule['applied_adjustments'])}")
            
            # AI-optimized
            if "error" not in results["ai_optimized"]:
                ai = results["ai_optimized"]
                print(f"  🧠 AI-optimized: ${ai['recommended_price']:.2f} ({ai['price_change_percent']:+.1f}%)")
                print(f"       Confidence: {ai['confidence']:.3f}, Revenue: ${ai['expected_revenue']:.2f}")
        else:
            print(f"❌ Error: {result['error']}")
    
    print(f"\n✅ API Demo completed!")


if __name__ == "__main__":
    demo_api()