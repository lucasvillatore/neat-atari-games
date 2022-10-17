import neat

class Pong:
    def __init__(self, game='Pong'):
        self.name = 'ALE/Pong-ram-v5'
        self.frameskip = 1
        self.my_rgb = None
        self.ball_rgb = [236, 236, 236]


    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'pong/configs/neat-config'
        )

        return neat_configuration

        blurred = cv2.GaussianBlur(state, (5, 5), 0)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        coordinates = []
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coordinates.append([cX,cY+35])
            except Exception as err:
                pass

        return coordinates