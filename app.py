from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle calculation helper
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    return angle if angle <= 180 else 360 - angle

def generate_frames(exercise_type):
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Define joints for angle calculation
                    if exercise_type == 'biceps':
                        joint_points = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]]
                    elif exercise_type == 'overhead':
                        joint_points = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]]
                    elif exercise_type == 'pushup':
                        joint_points = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]]
                    elif exercise_type == 'squat':
                        joint_points = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]]
                    else:
                        joint_points = []

                    if joint_points:
                        angle = calculate_angle(*joint_points)

                        # Count reps
                        if angle > 160:
                            stage = 'down'
                        if angle < 50 and stage == 'down':
                            stage = 'up'
                            counter += 1

                        # Display angle
                        cv2.putText(image, f'Angle: {int(angle)}', (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Display counter
                cv2.putText(image, f'Reps: {counter}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                _, buffer = cv2.imencode('.jpg', image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/biceps')
def biceps_feed():
    return Response(generate_frames('biceps'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/overhead')
def overhead_feed():
    return Response(generate_frames('overhead'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pushup')
def pushup_feed():
    return Response(generate_frames('pushup'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squat')
def squat_feed():
    return Response(generate_frames('squat'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
