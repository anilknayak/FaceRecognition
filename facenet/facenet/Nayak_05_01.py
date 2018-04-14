import sys
import real_time_face_recognition as rlfr

if __name__ == '__main__':
    rlfr.main(rlfr.parse_arguments(sys.argv[1:]))