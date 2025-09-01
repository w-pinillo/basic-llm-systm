

# This script is for the classification section of the system.py file

# Load the model
# Define get_completition_from_messages using local LLM

#Step 1 classification
delimiter = "###"

system_message = f"Email assistant {delimiter}"


user_message = f"""User input""" 
messages = [{'role':'system',
             'content': system_message},
            {'role':'user',
             'content': f"{delimiter}{user_message}{delimiter}"},
             ]

#response = get_completition_from_messages(messages)
#print(response)


#Example:
delimiter = "####"
system_message = f"""
You will be provided with customer service queries. The customer service query will be delimited with {delimiter} characters.
Classify each query into a primary category and a secondary category. 
Provide your output in json format with the keys: primary and secondary.

Primary categories: Billing, Technical Support, Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human
"""

user_message = f"""
I want you to delete my profile and all of my user data"""
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
] 
#response = get_completion_from_messages(messages)
#print(response)


