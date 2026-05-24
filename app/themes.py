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
            self.bg_color              = "#F8FAFC"
            self.secondary_bg_color    = "#F1F5F9"
            self.panel_bg_color        = "#FFFFFF"
            self.card_bg_color         = "#F1F5F9"
            self.primary_text_color    = "#0F172A"
            self.secondary_text_color  = "#475569"
            self.text_muted_color      = "#94A3B8"
            self.accent_color          = "#3B82F6"
            self.button_color          = "#3B82F6"
            self.button_hover_color    = "#2563EB"
            self.button_text_color     = "#FFFFFF"
            self.border_color          = "#E2E8F0"
            self.input_bg_color        = "#FFFFFF"
            self.disabled_color        = "#E2E8F0"
            self.disabled_text_color   = "#94A3B8"
            self.success_color         = "#10B981"
            self.warning_color         = "#F59E0B"
            self.error_color           = "#EF4444"
            self.light_text_color      = "#FFFFFF"
            self.font_family           = "Inter, Segoe UI, Arial"
            
        elif self.current_theme == "Dark":
            # Broadcast-style navy dark — matches main.py's BG/PANEL/CARD tokens
            self.bg_color              = "#0B0F1A"
            self.secondary_bg_color    = "#1C2438"   # CARD
            self.panel_bg_color        = "#141927"   # PANEL
            self.card_bg_color         = "#1C2438"
            self.primary_text_color    = "#F1F5F9"
            self.secondary_text_color  = "#94A3B8"
            self.text_muted_color      = "#475569"
            self.accent_color          = "#3B82F6"
            self.button_color          = "#3B82F6"
            self.button_hover_color    = "#2563EB"
            self.button_text_color     = "#FFFFFF"
            self.border_color          = "#2A3347"
            self.input_bg_color        = "#1C2438"
            self.disabled_color        = "#141927"
            self.disabled_text_color   = "#475569"
            self.success_color         = "#10B981"
            self.warning_color         = "#F59E0B"
            self.error_color           = "#EF4444"
            self.light_text_color      = "#FFFFFF"
            self.font_family           = "Inter, Segoe UI, Arial"
            
        elif self.current_theme == "Blue":
            self.bg_color              = "#0C1A2E"
            self.secondary_bg_color    = "#172E52"
            self.panel_bg_color        = "#112240"
            self.card_bg_color         = "#172E52"
            self.primary_text_color    = "#E0F2FE"
            self.secondary_text_color  = "#7DD3FC"
            self.text_muted_color      = "#38BDF8"
            self.accent_color          = "#38BDF8"
            self.button_color          = "#0EA5E9"
            self.button_hover_color    = "#0284C7"
            self.button_text_color     = "#FFFFFF"
            self.border_color          = "#1E3A5F"
            self.input_bg_color        = "#172E52"
            self.disabled_color        = "#112240"
            self.disabled_text_color   = "#38BDF8"
            self.success_color         = "#22C55E"
            self.warning_color         = "#FBBF24"
            self.error_color           = "#EF4444"
            self.light_text_color      = "#FFFFFF"
            self.font_family           = "Inter, Segoe UI, Arial"
            
        elif self.current_theme == "Green":
            self.bg_color              = "#052E16"
            self.secondary_bg_color    = "#14532D"
            self.panel_bg_color        = "#0D3B22"
            self.card_bg_color         = "#14532D"
            self.primary_text_color    = "#F0FDF4"
            self.secondary_text_color  = "#86EFAC"
            self.text_muted_color      = "#4ADE80"
            self.accent_color          = "#22C55E"
            self.button_color          = "#16A34A"
            self.button_hover_color    = "#15803D"
            self.button_text_color     = "#FFFFFF"
            self.border_color          = "#166534"
            self.input_bg_color        = "#14532D"
            self.disabled_color        = "#0D3B22"
            self.disabled_text_color   = "#4ADE80"
            self.success_color         = "#22C55E"
            self.warning_color         = "#FBBF24"
            self.error_color           = "#EF4444"
            self.light_text_color      = "#FFFFFF"
            self.font_family           = "Inter, Segoe UI, Arial"
            
        else:
            # Default to Light theme if an unknown theme is specified
            self.set_theme("Light")