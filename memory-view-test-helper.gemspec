lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "memory-view-test-helper/version"

clean_white_space = lambda do |entry|
  entry.gsub(/(\A\n+|\n+\z)/, "") + "\n"
end

Gem::Specification.new do |spec|
  spec.name = "memory-view-test-helper"
  spec.version = MemoryViewTestHelper::VERSION
  spec.homepage = "https://github.com/mrkn/memory-view-test-helper"
  spec.authors = ["Ketna Murata"]
  spec.email = ["mrkn@mrkn.jp"]

  readme = File.read("README.md")
  readme.force_encoding("UTF-8")
  entries = readme.split(/^\#\#\s(.*)$/)
  description = clean_white_space.call(entries[entries.index("Description") + 1])
  spec.summary, spec.description, = description.split(/\n\n+/, 3)
  spec.license = "MIT"
  spec.files = [
    "README.md",
    "LICENSE.txt",
    "Rakefile",
    "Gemfile",
    "#{spec.name}.gemspec",
  ]
  spec.files += Dir.glob("lib/**/*.rb")

  spec.add_development_dependency("bundler")
  spec.add_development_dependency("rake")
  spec.add_development_dependency("test-unit")
end
