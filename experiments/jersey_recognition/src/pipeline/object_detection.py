class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kits_clf=None
        self.left_team_label=0
        self.grass_hsv=None
    
    def detect(self, frame):
        """Returns structured detection results"""
        results = self.model(frame)[0]
        return {
            'boxes': results.boxes,
            'players': self._process_players(results),
            'teams': self._identify_teams(results)
        }

    def _get_players_boxes(self, results):
      """Extract player bounding boxes and images."""
      players_imgs = []
      players_boxes = []
      for box in results.boxes:
          label = int(box.cls.numpy()[0])
          if label == 0:  # Player
              x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
              player_img = results.orig_img[y1:y2, x1:x2]
              players_imgs.append(player_img)
              players_boxes.append(box)
      return players_imgs, players_boxes

    def _get_grass_color(self, img):
        """Detect the dominant grass color in the frame."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        grass_color = cv2.mean(img, mask=mask)
        return grass_color[:3]

    def _get_kits_colors(self, players, frame):
        """Extract kit colors for all players."""
        kits_colors = []
        if self.grass_hsv is None:
            grass_color = self._get_grass_color(frame)
            self.grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

        for player_img in players:
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([self.grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([self.grass_hsv[0, 0, 0] + 10, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask = cv2.bitwise_not(mask)
            upper_mask = np.zeros(player_img.shape[:2], np.uint8)
            upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
            mask = cv2.bitwise_and(mask, upper_mask)
            kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
            kits_colors.append(kit_color)
        return kits_colors

      def _get_kits_classifier(self, kits_colors):
        """Train a K-Means classifier to distinguish team kits."""
        kits_kmeans = KMeans(n_clusters=2)
        kits_kmeans.fit(kits_colors)
        return kits_kmeans

      def _classify_kits(self, kits_colors):
          """Classify players into teams based on kit colors."""
          return self.kits_clf.predict(kits_colors)

      def _identify_teams(self, results, frame):
        """Identify teams and their positions (left/right)."""
        players_imgs, players_boxes = self._get_players_boxes(results)
        kits_colors = self._get_kits_colors(players_imgs, frame)

        if self.kits_clf is None:
            self.kits_clf = self._get_kits_classifier(kits_colors)
            self.left_team_label = self._get_left_team_label(players_boxes, kits_colors)

        return {
            'kits_colors': kits_colors,
            'left_team_label': self.left_team_label
        }

      def _get_left_team_label(self, players_boxes, kits_colors):
        """Determine which team is on the left side of the frame."""
        team_0 = []
        team_1 = []

        for i in range(len(players_boxes)):
            x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())
            team = self._classify_kits([kits_colors[i]]).item()
            if team == 0:
                team_0.append(np.array([x1]))
            else:
                team_1.append(np.array([x1]))

        team_0 = np.array(team_0)
        team_1 = np.array(team_1)

        return 1 if np.average(team_0) - np.average(team_1) > 0 else 0
