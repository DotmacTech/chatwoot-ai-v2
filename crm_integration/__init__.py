"""
Mock CRM integration module for testing.
"""

def get_customer_info(customer_id):
    """Mock function to get customer info"""
    return {
        "id": customer_id,
        "name": "Test Customer",
        "email": "test@example.com",
        "phone": "123-456-7890",
        "account_type": "residential",
        "service_level": "premium"
    }

def update_customer_info(customer_id, data):
    """Mock function to update customer info"""
    return {"success": True, "customer_id": customer_id}

def find_customers_by_contact(contact_info):
    """Mock function to find customers by contact info"""
    return [{
        "id": "test123",
        "name": "Test Customer",
        "email": "test@example.com",
        "phone": "123-456-7890"
    }]
