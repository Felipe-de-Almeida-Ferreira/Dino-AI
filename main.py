from setup import *
from auxiliary import *

class Layer():
    def __init__(self, input_dim, output_dim, weights, bias, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = weights
        self.biases = bias
        self.activation = activation


        self._activ_inp, self._activ_out = None, None

class Dinossaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    
    def __init__(self):
        self.layers = []
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput, x):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >=10:
            self.step_index = 0
        
        if np.argmax(self.predict(x)) == 1 and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif np.argmax(self.predict(x)) == 2 and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif np.argmax(self.predict(x)) == 2 and not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1       
    
    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.dino_rect.y >= self.Y_POS:  # Garante que o dinossauro volte ao chão
            self.dino_rect.y = self.Y_POS
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def predict(self, x):
        return self.__feedforward(x)

    def __feedforward(self, x):
        self.layers[0].input = x
        for current_layer, next_layer in zip(self.layers, self.layers[1:] + [Layer(0, 0, 0, 0, 0)]):
            y = np.dot(current_layer.input, current_layer.weights.T) + current_layer.biases
            current_layer._activ_inp = y
            current_layer._activ_out = next_layer.input = current_layer.activation(y)
        return self.layers[-1]._activ_out

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))
        
class Obstacles:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -=game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacles):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325

class LargeCactus(Obstacles):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300

class Bird(Obstacles):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1

best = 0
gen_count = 0

def new_gen(weights_list, biases_list):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, x, b_weights, b_biases, gen_count
    run = True
    clock = pygame.time.Clock()
    cloud = Cloud()
    gen_count += 1
    game_speed = 14
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    players = []
    for i in range(50):
        weights = mutate_weights(weights_list[i])
        biases = mutate_biases(biases_list[i])
        #print(weights,'\n---------\n', biases, '\n|||||||||||||||||||||\n')
        
        dino = Dinossaur()
        dino.layers.append(Layer(input_dim=3, output_dim=3, weights=weights, bias=biases, activation=relu))
        dino.layers.append(Layer(input_dim=3, output_dim=3, weights=weights, bias=biases, activation=relu))
        dino.layers.append(Layer(input_dim=3, output_dim=3, weights=weights, bias=biases, activation=relu))
        players.append(dino)

    def score():
        global points, game_speed, best
        points += 1
        if points % 100 == 0:
            game_speed += 1
        if points >= best:
            best = points
        

        text = font.render("points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

        text = font.render("best: " + str(best), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (800, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        SCREEN.fill((255, 255, 255))
        userInput = pygame.key.get_pressed()

        text = font.render("gen: " + str(gen_count), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (100, 40)
        SCREEN.blit(text, textRect)


        if len(players) <= 1:
            if points > best:
                for layer in players[0].layers:
                    b_weights = layer.weights
                    b_biases = layer.biases
            
                new_gen(weights_list=[b_weights] * 20, biases_list=[b_biases] * 20)
            else:
                new_gen(weights_list=weights_list, biases_list=biases_list)
        '''for player in players:
            if player.dino_rect.y > 300:
                    players.remove(player)'''

        if len(obstacles) > 0:
            obstacle = obstacles[0]
            x = [game_speed, obstacle.rect.x, obstacle.rect.x]
        else:
            x = [game_speed, SCREEN_WIDTH, 0]

        
        for player in players:
            player.update(userInput, x)
            player.draw(SCREEN)

        if len(obstacles) == 0:
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            
            for player in players:
                if player.dino_rect.colliderect(obstacle.rect):
                        players.remove(player)

        background()
        
        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(30)
        pygame.display.update()
        
def main():
    weights_list = []
    biases_list = []

    for i in range(50):
        weight_set = random_normal(3, 3)
        biases_set = ones(3, 3)

        weights_list.append(weight_set)
        biases_list.append(biases_set)


    new_gen(weights_list=weights_list, biases_list=biases_list)

main()