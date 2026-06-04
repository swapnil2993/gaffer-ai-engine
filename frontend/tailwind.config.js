/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'football-green': '#0d1b15',
        'pitch-accent': '#1a2e26',
        'scout-gold': '#d4af37',
      },
    },
  },
  plugins: [],
}
