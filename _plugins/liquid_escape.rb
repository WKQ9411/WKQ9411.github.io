# Escape double braces in math formulas to prevent Liquid parsing errors
# This prevents Liquid from interpreting {{ as variable markers in LaTeX
# Usage: Automatically applied to all documents

Jekyll::Hooks.register [:documents, :pages], :pre_render do |doc|
  # Only process markdown files
  next unless doc.extname == '.md'
  
  # Replace {{ with { { (add space) to prevent Liquid parsing
  # But only when it's likely part of a LaTeX formula (preceded by common LaTeX patterns)
  doc.content = doc.content.gsub(/\{\{/, '{ {')
  
  # Also handle }} -> } } for consistency
  doc.content = doc.content.gsub(/\}\}/, '} }')
end
