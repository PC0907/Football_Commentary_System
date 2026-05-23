class ThemeManager:
    """
    Manages application themes with consistent color schemes and styles.
    Themes can be easily switched without affecting the application's functionality.
    """
    
    def __init__(self, initial_theme="Light"):
        self.current_theme = initial_theme
        self.update_theme_values()
    
    def set_theme(self, theme_name):
        """Change the current theme"""
        self.current_theme = theme_name
        self.update_theme_values()
    
    def update_theme_values(self):
        """Update theme values based on the current theme"""
        if self.current_theme == "Light":
            self.bg_color = "#f5f5f5"
            self.secondary_bg_color = "#e0e0e0"
            self.primary_text_color = "#212121"
            self.secondary_text_color = "#757575"
            self.accent_color = "#2196F3"
            self.button_color = "#2196F3"
            self.button_hover_color = "#1976D2"
            self.button_text_color = "#FFFFFF"
            self.border_color = "#BDBDBD"
            self.input_bg_color = "#FFFFFF"
            self.disabled_color = "#BDBDBD"
            self.disabled_text_color = "#757575"
            self.success_color = "#4CAF50"
            self.warning_color = "#FFC107"
            self.error_color = "#F44336"
            self.light_text_color = "#FFFFFF"
            self.font_family = "Arial"
            
        elif self.current_theme == "Dark":
            self.bg_color = "#212121"
            self.secondary_bg_color = "#303030"
            self.primary_text_color = "#EEEEEE"
            self.secondary_text_color = "#AAAAAA"
            self.accent_color = "#2196F3"
            self.button_color = "#2196F3"
            self.button_hover_color = "#1976D2"
            self.button_text_color = "#FFFFFF"
            self.border_color = "#424242"
            self.input_bg_color = "#424242"
            self.disabled_color = "#424242"
            self.disabled_text_color = "#757575"
            self.success_color = "#4CAF50"
            self.warning_color = "#FFC107"
            self.error_color = "#F44336"
            self.light_text_color = "#FFFFFF"
            self.font_family = "Arial"
            
        elif self.current_theme == "Blue":
            self.bg_color = "#E3F2FD"
            self.secondary_bg_color = "#BBDEFB"
            self.primary_text_color = "#0D47A1"
            self.secondary_text_color = "#1565C0"
            self.accent_color = "#2196F3"
            self.button_color = "#1976D2"
            self.button_hover_color = "#0D47A1"
            self.button_text_color = "#FFFFFF"
            self.border_color = "#90CAF9"
            self.input_bg_color = "#FFFFFF"
            self.disabled_color = "#BBDEFB"
            self.disabled_text_color = "#64B5F6"
            self.success_color = "#00C853"
            self.warning_color = "#FFD600"
            self.error_color = "#D50000"
            self.light_text_color = "#FFFFFF"
            self.font_family = "Arial"
            
        elif self.current_theme == "Green":
            self.bg_color = "#E8F5E9"
            self.secondary_bg_color = "#C8E6C9"
            self.primary_text_color = "#1B5E20"
            self.secondary_text_color = "#2E7D32"
            self.accent_color = "#4CAF50"
            self.button_color = "#388E3C"
            self.button_hover_color = "#1B5E20"
            self.button_text_color = "#FFFFFF"
            self.border_color = "#A5D6A7"
            self.input_bg_color = "#FFFFFF"
            self.disabled_color = "#C8E6C9"
            self.disabled_text_color = "#81C784"
            self.success_color = "#00C853"
            self.warning_color = "#FFD600"
            self.error_color = "#D50000"
            self.light_text_color = "#FFFFFF"
            self.font_family = "Arial"
            
        else:
            # Default to Light theme if an unknown theme is specified
            self.set_theme("Light")