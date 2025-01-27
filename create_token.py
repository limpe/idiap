from telegraph import Telegraph
from typing import Optional

def create_telegraph_token() -> Optional[str]:
    """
    Create a Telegraph account and get access token.
    Run this script once to get your token.
    """
    try:
        telegraph = Telegraph()
        
        # Create account
        account = telegraph.create_account(
            short_name='Paidi',
            author_name='Paidi Analysis Bot',
            author_url='https://t.me/your_bot_username'
        )
        
        # Get and print the access token
        token = account['access_token']
        print("\nYour Telegraph Access Token:")
        print("============================")
        print(token)
        print("\nSave this token and add it to your environment variables as TELEGRAPH_TOKEN")
        print("The token won't expire, so you can keep using it.")
        
        return token
    except Exception as e:
        print(f"Error creating Telegraph account: {e}")
        return None

if __name__ == "__main__":
    create_telegraph_token()