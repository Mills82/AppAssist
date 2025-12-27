/* clients/web/tailwind.config.js
 * Tailwind CSS configuration for the clients/web Next.js App Router.
 * Scans app/, components/ (and pages/ for compatibility) for utility class usage.
 */

module.exports = {
  content: [
    "./app/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}",
    "./pages/**/*.{js,jsx,ts,tsx}"
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#2563eb",
          50: "#eef6ff",
          100: "#dbeeff",
          200: "#b6dbff",
          300: "#8fc7ff",
          400: "#5aaeff",
          500: "#2563eb",
          600: "#1f4fc0",
          700: "#193b96",
          800: "#12286b",
          900: "#0b153f"
        }
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Helvetica Neue", "Arial"]
      }
    }
  },
  plugins: []
};
