/** Build-time config for the vendored stylesheet (see static/README or
 *  windows/build_exe.ps1). Mirrors what cdn.tailwindcss.com did at
 *  runtime: stock v3 theme, no plugins, scanning the templates and the
 *  Python file that emits markup. */
module.exports = {
  content: ["./templates/**/*.html", "./webapp.py"],
  theme: { extend: {} },
  plugins: [],
};
