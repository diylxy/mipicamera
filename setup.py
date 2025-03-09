from distutils.core import setup, Extension
import numpy
def main():
    setup(name="mipicamera",
          version="1.0.0",
          description="Python interface for the fputs C library function",
          author="<your name>",
          author_email="your_email@gmail.com",
          ext_modules=[
              Extension(
                  "mipicamera", ["mipicamera.c", "v4l2-camera.c"], 
                  extra_link_args = ['-lm'],
                  define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                  include_dirs=[numpy.get_include()],
          )])

if __name__ == "__main__":
    main()
