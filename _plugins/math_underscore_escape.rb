# Escape underscores, asterisks and pipes in inline math formulas ($...$) before kramdown processing
# This prevents kramdown from interpreting these characters as markdown syntax
# Usage: Automatically applied to all documents

Jekyll::Hooks.register [:documents, :pages], :pre_render do |doc|
  # Only process markdown files
  next unless doc.extname == '.md'
  
  # Regex to match inline math: $...$ but not $$...$$
  # We need to escape underscores, asterisks and pipes inside inline math
  doc.content = doc.content.gsub(/(?<!\$)\$(?!\$)([^\$]+?)\$(?!\$)/) do |match|
    inner = $1
    # Escape underscores that are not already escaped
    escaped_inner = inner.gsub(/(?<!\\)_/, '\\_')
    # Escape asterisks that are not already escaped
    escaped_inner = escaped_inner.gsub(/(?<!\\)\*/, '\\*')
    # Escape pipes that are not already escaped (prevents table parsing)
    escaped_inner = escaped_inner.gsub(/(?<!\\)\|/, '\\|')
    "$#{escaped_inner}$"
  end
end
