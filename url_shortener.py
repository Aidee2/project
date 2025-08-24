import string
import random
import json
import os
from datetime import datetime
from urllib.parse import urlparse
import webbrowser

class URLShortener:
    def __init__(self, data_file='url_data.json'):
        self.data_file = data_file
        self.url_database = {}
        self.analytics = {}
        self.load_data()

    def load_data(self):
        """Load existing URLs and analytics from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.url_database = data.get('urls', {})
                    self.analytics = data.get('analytics', {})
            except:
                self.url_database = {}
                self.analytics = {}
    
    def save_data(self):
        """Save URLs and analytics to file"""
        data = {
            'urls': self.url_database,
            'analytics': self.analytics
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def generate_short_code(self, length=6):
        """Generate random short code"""
        characters = string.ascii_letters + string.digits
        while True:
            code = ''.join(random.choice(characters) for _ in range(length))
            if code not in self.url_database:
                return code
    
    def shorten_url(self, original_url, custom_code=None):
        """Shorten a URL"""
        # Add http:// if missing
        if not original_url.startswith(('http://', 'https://')):
            original_url = 'http://' + original_url
        
        # Validate URL
        if not self.is_valid_url(original_url):
            return None, "Invalid URL format"
        
        # Check if URL already exists
        for code, data in self.url_database.items():
            if data['original_url'] == original_url:
                return code, "URL already exists"
        
        # Generate or use custom code
        if custom_code:
            if custom_code in self.url_database:
                return None, "Custom code already exists"
            short_code = custom_code
        else:
            short_code = self.generate_short_code()
        
        # Store URL data
        self.url_database[short_code] = {
            'original_url': original_url,
            'created_at': datetime.now().isoformat(),
            'clicks': 0
        }
        
        # Initialize analytics
        self.analytics[short_code] = {
            'total_clicks': 0,
            'click_history': []
        }
        
        self.save_data()
        return short_code, "URL shortened successfully"
    
    def expand_url(self, short_code):
        """Get original URL from short code"""
        if short_code in self.url_database:
            # Update click count
            self.url_database[short_code]['clicks'] += 1
            self.analytics[short_code]['total_clicks'] += 1
            self.analytics[short_code]['click_history'].append({
                'timestamp': datetime.now().isoformat()
            })
            self.save_data()
            
            return self.url_database[short_code]['original_url']
        return None
    
    def get_analytics(self, short_code):
        """Get analytics for a short code"""
        if short_code in self.url_database:
            url_data = self.url_database[short_code]
            analytics_data = self.analytics.get(short_code, {})
            
            return {
                'short_code': short_code,
                'original_url': url_data['original_url'],
                'created_at': url_data['created_at'],
                'total_clicks': analytics_data.get('total_clicks', 0),
                'click_history': analytics_data.get('click_history', [])
            }
        return None
    
    def list_all_urls(self):
        """List all shortened URLs"""
        return [
            {
                'short_code': code,
                'original_url': data['original_url'],
                'clicks': data['clicks'],
                'created_at': data['created_at']
            }
            for code, data in self.url_database.items()
        ]

# Command Line Interface
def command_line_interface():
    shortener = URLShortener()
    
    print("="*50)
    print("         URL SHORTENER")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Shorten a URL")
        print("2. Expand a short URL")
        print("3. View analytics")
        print("4. List all URLs")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            url = input("Enter URL to shorten: ").strip()
            custom = input("Custom code (optional, press Enter to skip): ").strip()
            custom = custom if custom else None
            
            code, message = shortener.shorten_url(url, custom)
            if code:
                print(f"\n✅ {message}")
                print(f"Short URL: http://localhost:5000/{code}")
                print(f"Short Code: {code}")
            else:
                print(f"\n❌ Error: {message}")
        
        elif choice == '2':
            code = input("Enter short code: ").strip()
            original = shortener.expand_url(code)
            if original:
                print(f"\n✅ Original URL: {original}")
                print("Opening in browser...")
                webbrowser.open(original)
            else:
                print("\n❌ Short code not found")
        
        elif choice == '3':
            code = input("Enter short code for analytics: ").strip()
            analytics = shortener.get_analytics(code)
            if analytics:
                print(f"\n📊 Analytics for {code}:")
                print(f"Original URL: {analytics['original_url']}")
                print(f"Created: {analytics['created_at']}")
                print(f"Total clicks: {analytics['total_clicks']}")
            else:
                print("\n❌ Short code not found")
        
        elif choice == '4':
            urls = shortener.list_all_urls()
            if urls:
                print(f"\n📋 All URLs ({len(urls)} total):")
                print("-" * 80)
                for url in urls:
                    print(f"Code: {url['short_code']} | Clicks: {url['clicks']} | URL: {url['original_url'][:50]}...")
            else:
                print("\n📋 No URLs found")
        
        elif choice == '5':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    command_line_interface()
