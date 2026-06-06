project('simplex', ['c', 'cpp'])

cc = meson.get_compiler('cpp')
umfpack_dep = dependency('UMFPACK', required: true)

amd_dep = cc.find_library('amd', required: true)
suitesparseconfig_dep = cc.find_library('suitesparseconfig', required: true)

includes = ['thirdparty/']

sources = []
subdir('src')

executable('simplex',
  sources,
  include_directories: includes,

  dependencies: [
    umfpack_dep,
    amd_dep,
    suitesparseconfig_dep
  ]
)
