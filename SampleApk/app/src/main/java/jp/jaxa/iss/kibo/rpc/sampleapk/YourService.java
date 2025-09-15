package jp.jaxa.iss.kibo.rpc.sampleapk;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.Interpreter;

import gov.nasa.arc.astrobee.Kinematics;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

/*
 (NOTE) target have to be taken within 30deg and 0.9m
 */
public class YourService extends KiboRpcService {

    // Mapping of item, checking for treasure items
    public static final Map<String, Integer> item_map;
    static {
        item_map = new HashMap<>();
        item_map.put("crystal", 1);
        item_map.put("diamond", 1);
        item_map.put("emerald", 1);
        item_map.put("coin", 0);
        item_map.put("compass", 0);
        item_map.put("coral", 0);
        item_map.put("fossil", 0);
        item_map.put("key", 0);
        item_map.put("letter", 0);
        item_map.put("shell", 0);
        item_map.put("treasure_box", 0);
    }

    // YOLOv8 TFLite Interpreter & config
    private Interpreter tflite;
//    private static final int INPUT_SIZE = 640;
    private static final int INPUT_SIZE = 768;
    private static final int NUM_THREADS = 4;
    private static final float CONF_THRESHOLD = 0.4f;
    private static final String[] classNames = {
            "coin","compass","coral","crystal","diamond",
            "emerald","fossil","key","letter","shell","treasure_box"
    };
    private Mat[] cropIMG = new Mat[5];
    private Mat cropPaper[] = new Mat[5];
    private int[][] areaItemCount = new int[5][12];
    private Mat[] fullImages = new Mat[5];
    private Mat[] cameraMatrix = new Mat[5];
    private MatOfDouble[] distCoeffs = new MatOfDouble[5];
    private Kinematics[] kinArr = new Kinematics[5];
    private Point[] points = {
            new Point(10.95f, -9.85f, 5.195f),
            new Point(10.925f, -8.875f, 4.995f),
            new Point(10.925f, -7.925f, 4.995f),
            new Point(11.175f, -6.875f, 4.685f),
            new Point(11.143f, -6.7607f, 4.9654f)
    };
    private Quaternion[] quats = {
//            QuaternionUtils.multiply(
//                    QuaternionUtils.fromEulerDegrees(0f,-20f,0f),
//                    QuaternionUtils.fromEulerDegrees(0f,0f,-90f)),
            QuaternionUtils.fromEulerDegrees(0f,0f,-90f),
            QuaternionUtils.fromEulerDegrees(0f,90f,0f),
            QuaternionUtils.fromEulerDegrees(0f,90f,0f),
            QuaternionUtils.fromEulerDegrees(0f,0f,180f),
            QuaternionUtils.fromEulerDegrees(0f,0f,90f)
    };

    // Copy asset file to a temp file so Interpreter can read it.
    private File convertModelFileFromAssetsToTempFile(String modelFileName) throws IOException {
        InputStream is = getAssets().open(modelFileName);
        File tmp = File.createTempFile(modelFileName, null, getCacheDir());
        tmp.deleteOnExit();
        try (FileOutputStream os = new FileOutputStream(tmp)) {
            byte[] buf = new byte[4096];
            int len;
            while ((len = is.read(buf)) > 0) {
                os.write(buf, 0, len);
            }
        }
        return tmp;
    }

    // Initialize Interpreter
    private void initInterpreter() throws IOException {
        if (tflite != null) return;
        File modelFile = convertModelFileFromAssetsToTempFile("best.tflite");
        try (FileInputStream fis = new FileInputStream(modelFile)) {
            MappedByteBuffer buf = fis.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, fis.getChannel().size());
            Interpreter.Options opts = new Interpreter.Options().setNumThreads(NUM_THREADS);
            tflite = new Interpreter(buf, opts);
        }
    }

    // Preprocess Mat -> ByteBuffer
    private ByteBuffer preprocess(Mat matImage) {
        Bitmap bmp = Bitmap.createBitmap(matImage.cols(), matImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matImage, bmp);
        Bitmap resized = Bitmap.createScaledBitmap(bmp, INPUT_SIZE, INPUT_SIZE, true);

        ByteBuffer buf = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4).order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int p : pixels) {
            float r = ((p >> 16) & 0xFF)/255f;
            float g = ((p >> 8) & 0xFF)/255f;
            float b = (p & 0xFF)/255f;
            buf.putFloat(r).putFloat(g).putFloat(b);
        }
        buf.rewind();
        return buf;
    }

    // Detection using Interpreter
    public List<DetectionResult> detectItems(int imageNum, Mat matImage) throws IOException {
        initInterpreter();
        // Prepare a debug clone
        Mat debugMat = matImage.clone();
        int origW = matImage.cols();
        int origH = matImage.rows();
        ByteBuffer input = preprocess(matImage);

        // Run inference
        final int MAX_DET = 20;
        float[][][] output = new float[1][MAX_DET][6];
        tflite.run(input, output);

        List<DetectionResult> found = new ArrayList<>();

        // BGR colors: red box, red background, white text
        Scalar boxColor = new Scalar(0, 0, 255);  // red
        Scalar labelBgColor = new Scalar(0, 0, 255);  // red fill
        Scalar labelTextColor = new Scalar(255, 255, 255); // white

        Map<List<Integer>, Boolean> pointMap = new HashMap<>();

        for (int i = 0; i < MAX_DET; i++) {
            float x1n = output[0][i][0];
            float y1n = output[0][i][1];
            float x2n = output[0][i][2];
            float y2n  = output[0][i][3];
            float score = output[0][i][4];
            int cls = (int)output[0][i][5];

            if (score < CONF_THRESHOLD) continue;

            // Denormalize
            int x1 = (int)(x1n * origW), y1 = (int)(y1n * origH);
            int x2 = (int)(x2n * origW), y2 = (int)(y2n * origH);
            x1 = Math.max(0, Math.min(x1, origW));
            y1 = Math.max(0, Math.min(y1, origH));
            x2 = Math.max(0, Math.min(x2, origW));
            y2 = Math.max(0, Math.min(y2, origH));

            List<Integer> pointList = new ArrayList<>();
            pointList.add(x1);
            pointList.add(y1);
            pointList.add(x2);
            pointList.add(y2);
            if(pointMap.containsKey(pointList)) continue;
            pointMap.put(pointList, true);

            //Draw box
            Imgproc.rectangle(debugMat,
                    new org.opencv.core.Point(x1, y1),
                    new org.opencv.core.Point(x2, y2),
                    boxColor,
                    2);

            //Label background
            String label = classNames[cls] + String.format(" %.2f", score);
            int[] baseLine = new int[1];
            Size ts = Imgproc.getTextSize(label,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, 2, baseLine);

            Imgproc.rectangle(debugMat,
                    new org.opencv.core.Point(x1, y1 - ts.height - baseLine[0]),
                    new org.opencv.core.Point(x1 + ts.width, y1),
                    labelBgColor,
                    Imgproc.FILLED);

            //Label text
            Imgproc.putText(debugMat,
                    label,
                    new org.opencv.core.Point(x1, y1 - baseLine[0]),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    labelTextColor,
                    2);

            found.add(new DetectionResult(classNames[cls], score));
            Log.i("Detection", String.format("Img%d: %s score=%.2f box=[%d,%d,%d,%d]", imageNum, classNames[cls], score, x1, y1, x2, y2));
        }

        // Save debug image
        api.saveMatImage(debugMat, "debug_items_" + imageNum + ".png");
        return found;
    }

    public Mat image_cropping(Mat matImage, int partsX, int partsY, int startX, int startY, int scaleX, int scaleY) {
        Mat image = matImage.clone();
        int width = image.cols(), height = image.rows();
        int cellW = width / partsX, cellH = height / partsY;
        int cropW = cellW * scaleX, cropH = cellH * scaleY;
        int x = startX * cellW, y = startY * cellH;
        if (startX+scaleX>partsX||startY+scaleY>partsY||x<0||y<0||x+cropW>width||y+cropH>height) {
            return image;
        }
        return new Mat(image, new Rect(x,y,cropW,cropH));
    }

    public Mat undistorting(Mat image) {
        double[][] cam = api.getNavCamIntrinsics();
        Mat mtx = new Mat(3,3,CvType.CV_32FC1);
        mtx.put(0,0,cam[0]);
        Mat dc = new Mat(1,5,CvType.CV_32FC1);
        dc.put(0,0,cam[1]);
        Mat und = new Mat();
        Calib3d.undistort(image,und,mtx,dc);
        Mat sharp = new Mat();
        Mat k = new Mat(3,3,CvType.CV_32FC1);
        k.put(0,0,0,-1,0,-1,5,-1,0,-1,0);
        Imgproc.filter2D(und,sharp,-1,k);
        return sharp;
    }

    public Mat arCropping(Mat image) {
        Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>(); Mat ids = new Mat();
        Aruco.detectMarkers(image,dict,corners,ids);
        if (corners.isEmpty()) return null;
        MatOfPoint2f src = new MatOfPoint2f(corners.get(0));
        int size = 200;
        MatOfPoint2f dst = new MatOfPoint2f(
                new org.opencv.core.Point(0,0),
                new org.opencv.core.Point(size,0),
                new org.opencv.core.Point(size,size),
                new org.opencv.core.Point(0,size)
        );
        Mat warp = Imgproc.getPerspectiveTransform(src,dst);
        Mat out  = new Mat();
        Imgproc.warpPerspective(image,out,warp,new org.opencv.core.Size(size,size));
        return out;
    }

    public Mat getNavCam() {
        return undistorting(api.getMatNavCam());
    }

    public Mat cropAroundMarker(int area, Mat image, double expandScale) {
        Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(image, dict, corners, ids);
        int nAR = ids.rows();
        if (nAR==0){
            Log.e("CropAroundMakers", "AR at area " + area + " is not found.");
            return null;
        }
        int use = 0;
        if(nAR>1){
            double[] cx = new double[nAR];
            for(int i = 0; i < nAR; i++){
                MatOfPoint2f mf = new MatOfPoint2f(corners.get(i));
                org.opencv.core.Point[] pts = mf.toArray();
                double sumX = 0;
                for(org.opencv.core.Point p: pts) sumX += p.x;
                cx[i] = sumX / pts.length;
            }
            if(area==1) {
                if(cx[0]>cx[1]) use = 1;
                else use = 0;
            } else if(area==2) {
                if(cx[0]>cx[1]) use = 0;
                else use = 1;
            } else {
                Log.w("AR", "Found more than 2 ar in image " + area);
            }
        }

        //Take the first marker’s 4 corner points
        MatOfPoint2f markerPts2f = new MatOfPoint2f(corners.get(use));
        org.opencv.core.Point[] pts = markerPts2f.toArray();

        //Build an integer-point array for boundingRect
        org.opencv.core.Point[] ptsIntArr = new org.opencv.core.Point[pts.length];
        for (int j = 0; j < pts.length; j++) {
            // round to nearest integer
            int x = (int) Math.round(pts[j].x);
            int y = (int) Math.round(pts[j].y);
            ptsIntArr[j] = new org.opencv.core.Point(x, y);
        }
        MatOfPoint ptsInt = new MatOfPoint(ptsIntArr);

        // Compute the axis-aligned bounding rect of those 4 points
        Rect bbox = Imgproc.boundingRect(ptsInt);

        // Expand that box around its center
        int w = bbox.width, h = bbox.height;
        int newW = (int) (w * expandScale);
        int newH = (int) (h * expandScale);
        int cx = bbox.x + w / 2;
        int cy = bbox.y + h / 2;

        int x1 = cx - newW / 2;
        int y1 = cy - newH / 2;
        // clamp to image borders
        x1 = Math.max(0, x1);
        y1 = Math.max(0, y1);
        int x2 = Math.min(image.cols(), x1 + newW);
        int y2 = Math.min(image.rows(), y1 + newH);

        // Return the cropped region
        return new Mat(image, new Rect(x1, y1, x2 - x1, y2 - y1));
    }

    public float clamp(float min, float value, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static final float MARKER_LEN = 0.05f;   // 5 cm
    private static final float MAX_DIST = 0.9f;    // 0.9 m
    private static final float MAX_ANGLE = 30f;     // 30°
    private static final float OFFSET = 0.75f;

    public Mat approachAndSnapshot(int k) throws IOException {
        if (k < 0 || k >= 4) {
            Log.e("Nav", "invalid index " + k);
            return null;
        }

        // retrieve saved image, intrinsics, and kinematics
        Mat img = fullImages[k];
        Mat K = cameraMatrix[k];
        MatOfDouble D = distCoeffs[k];
        Kinematics kin = kinArr[k];
        Point basePos = kin.getPosition();
        Quaternion baseQ = kin.getOrientation();

        Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        DetectorParameters params = DetectorParameters.create();
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(img, dict, corners, ids, params);
        if (ids.empty()) {
            Log.e("Nav", "Lost AR on snapshot image!");
            return null;
        }

        Mat rvecs = new Mat(), tvecs = new Mat();
        Aruco.estimatePoseSingleMarkers(corners, MARKER_LEN, K, D, rvecs, tvecs);
        if (tvecs.empty()) {
            Log.e("Nav", "Pose estimation failed");
            return null;
        }

        int n = tvecs.rows(); // n<=2
        double[] rel = new double[n];
        for (int i = 0; i < n; i++) {
            double[] tv = tvecs.row(i).get(0, 0);
            rel[i] = tv[0];
            Log.i("Snapshot", String.format("Tvecs[%d]: %s", i, tvecs.row(i).get(0, 0)));
        }

        int use = 0;
        if (n > 1) {
            if(k==1) {
                if (rel[0] < rel[1]) use = 0;
                else use = 1;
            } else if(k==2) {
                if(rel[0] < rel[1]) use = 1;
                else use = 0;
            } else {
                Log.w("Snapshot", "n > 1 and not in area 1, 2. Something is wrong.");
            }
        }

        Mat tvec = tvecs.row(use);
        double[] tv = tvec.get(0, 0);

        Point targetPos = new Point();
        Quaternion targetQuat = new Quaternion();
        float pX = (float)basePos.getX();
        float pY = (float)basePos.getY();
        float pZ = (float)basePos.getZ();
//        +Z is in front of the camera, +X is to the right, and +Y is down.
        if(k==0){
            targetPos = new Point(clamp(10.4f, (float)(pX + tv[0]), 11.45f),
                    clamp(-10.1f, (float)(-10.58 + OFFSET), -6.1f),
                    clamp(4.42f, (float)(pZ + tv[1]), 5.47f));
        } else if(k==1){
//            targetPos = new Point(pX + tv[1], pY + tv[0], 3.76093 + OFFSET);
            targetPos = new Point(clamp(10.4f, (float)(pX + tv[1]), 11.45f),
                    clamp(-10.1f, (float)(pY + tv[0]), -6.1f),
                    clamp(4.42f, (float)(3.76093 + OFFSET), 5.47f));
        } else if(k==2){
//            targetPos = new Point(pX + tv[1], pY + tv[0], 3.76093 + OFFSET);
            targetPos = new Point(clamp(10.4f, (float)(pX + tv[1]), 11.45f),
                    clamp(-10.1f, (float)(pY + tv[0]), -6.1f),
                    clamp(4.42f, (float)(3.76093 + OFFSET), 5.47f));
        } else {
//            targetPos = new Point(9.866984 + OFFSET, pY - tv[0], pZ + tv[1]);
            targetPos = new Point(clamp(10.4f, (float)(9.866984 + OFFSET), 11.45f),
                    clamp(-10.1f, (float)(pY - tv[0]), -6.1f),
                    clamp(4.42f, (float)(pZ + tv[1]), 5.47f));
        }
        targetQuat = quats[k];
        Log.i("Nav", String.format("Robot saved position: pos=(%.2f, %.2f, %.2f)", pX, pY, pZ));
        Log.i("Nav", String.format("Tvec: (%.2f, %.2f, %.2f)", tv[0], tv[1], tv[2]));
        Log.i("Nav", String.format("Moving to target: pos=%s, ori=%s", targetPos, targetQuat));
        api.moveTo(targetPos, targetQuat, false);
        SystemClock.sleep(3000);
        Mat closeImage = getNavCam();
        return closeImage;
    }

    private Mat getCameraMatrix() {
        double[][] cam = api.getNavCamIntrinsics();
        Mat K = new Mat(3,3, CvType.CV_64F);
        K.put(0,0, cam[0]);
        return K;
    }

    private MatOfDouble getDistCoeffs() {
        double[][] cam = api.getNavCamIntrinsics();
        return new MatOfDouble(cam[1]);
    }

    private int getItemIndex(String s) {
        for(int i=0;i<11;i++) if(s==classNames[i]) return i;
        return -1;
    }

    @Override
    protected void runPlan1() {
        api.startMission();
        for(int i=0;i<4;i++){
            api.moveTo(points[i], quats[i], false);
            SystemClock.sleep(3300);
            fullImages[i] = getNavCam();
            cameraMatrix[i] = getCameraMatrix();
            distCoeffs[i] = getDistCoeffs();
            kinArr[i] = api.getRobotKinematics();
            kinArr[i] = api.getRobotKinematics();
            api.saveMatImage(fullImages[i], "fullImage" + i + ".png");

            if(i==0) {
//                cropIMG[0] = image_cropping(fullImages[0], 16, 2, 1, 1, 14, 1);
                cropIMG[0] = fullImages[0].clone();
                cropPaper[0] = cropAroundMarker(i, cropIMG[0], 10.5);
            } else if(i==1) {
                cropIMG[1] = image_cropping(fullImages[1], 4, 1, 1, 0, 2, 1);
                cropPaper[1] = cropAroundMarker(i, cropIMG[1], 10);
            } else if(i==2) {
                cropIMG[2] = image_cropping(fullImages[2], 4, 1, 1, 0, 2, 1);
                cropPaper[2] = cropAroundMarker(i, cropIMG[2], 10);
            } else {
                cropIMG[3] = image_cropping(fullImages[3], 4, 1, 1, 0, 2, 1);
                cropPaper[3] = cropAroundMarker(i, cropIMG[3], 10);
            }
            api.saveMatImage(cropIMG[i], "croppedImage" + i + ".png");
            api.saveMatImage(cropPaper[i], "CroppedPaper"+ i +".png");


//            Mat temp = warpDocument(fullImages[i]);
            String name = "shell"; // By default, maybe can get some case correct 555
            int count = 0;
            List<DetectionResult> found = new ArrayList<>();
            try {
//                found = detectItems(i, warpDocument(cropPaper[i]));
                if(cropPaper[i]!=null) found = detectItems(i, cropPaper[i]);
                for(DetectionResult iter : found){
                    String item = iter.name;
                    if (item_map.containsKey(item)) {
                        if(item_map.get(item) == 0){
                            count++;
                            name = item;
                        }
                        areaItemCount[i][getItemIndex(item)] += 1;
                    }
                }
                if(found.isEmpty() || count==0){
                    Log.i("Detection", "Cropped Paper " + i + " not found. Using moving to area to close snapshot to process.");
                    for(int it=0;it<11;it++) areaItemCount[i][it] = 0;
                    name = "shell"; // By default, maybe can get some case correct 555
                    count = 0;
//                    found = detectItems(i, cropPaper[i]);
                    Mat closeImage = approachAndSnapshot(i);
                    api.saveMatImage(closeImage, "closeSnapshot"+i+".png");
                    Mat croppedCloseImage = cropAroundMarker(i, closeImage, 10);
                    if(croppedCloseImage==null) croppedCloseImage = closeImage;
                    api.saveMatImage(croppedCloseImage, "croppedCloseSnapshot" + i + ".png");
                    found = detectItems(i, croppedCloseImage);
                    for(DetectionResult iter : found){
                        String item = iter.name;
                        if (item_map.containsKey(item)) {
                            if(item_map.get(item) == 0){
                                count++;
                                name = item;
                            }
                            areaItemCount[i][getItemIndex(item)] += 1;
                        }
                    }
                }
                int mx = 0, pos = -1;
                for(int j=0;j<11;j++){
                    if(item_map.get(classNames[j])==0 && areaItemCount[i][j]>mx){
                        mx = areaItemCount[i][j];
                        pos = j;
                    }
                }
                if(pos!=-1){
                    name = classNames[pos];
                    count = mx;
                }
            } catch(IOException e){
                e.printStackTrace();
            }
            api.setAreaInfo(i+1, name, count);
        }

        api.moveTo(points[4], quats[4], false);
        api.reportRoundingCompletion();
        api.notifyRecognitionItem();
        SystemClock.sleep(3000);
        fullImages[4] = getNavCam();
        cropIMG[4] = image_cropping(fullImages[4], 19, 14, 7, 5, 6, 4);
//        cropPaper[4] = cropAroundMarker(cropIMG[4], 6);
        api.saveMatImage(fullImages[4], "fullImage4.png");
        api.saveMatImage(cropIMG[4], "croppedImage4.png");
        for(int i=0;i<5;i++) api.saveMatImage(arCropping(cropIMG[i]), "AR_CroppedImage" + i + ".png");
        String targetName = "diamond";
        int count = 0;
        float max_conf = -1f;
        List<DetectionResult> found = new ArrayList<>();
        try {
//            Mat temp = warpDocument(fullImages[4]);
            if(cropIMG[4]!=null) found = detectItems(4, cropIMG[4]);
            for(DetectionResult iter : found){
                String item = iter.name;
                float conf = iter.conf;
                if (item_map.containsKey(item)) {
                    if(item_map.get(item) == 1){
                        if(conf>max_conf) {
                            targetName = item;
                            max_conf = conf;
                        }
                        count++;
                    }
                }
            }
            if(found.isEmpty() || count==0){
                Log.i("Detection", "Cropped Image " + 4 + " item not found. Using croppedImageArea to process.");
                count = 0;
                max_conf = -1f;
                found = detectItems(4, fullImages[4]);
                for(DetectionResult iter : found){
                    String item = iter.name;
                    float conf = iter.conf;
                    if (item_map.containsKey(item)) {
                        if(item_map.get(item) == 1){
                            count++;
                            if(conf>max_conf) {
                                targetName = item;
                                max_conf = conf;
                            }
                        }
                    }
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
        //Walking to target item to take snapshot
        int tarId = getItemIndex(targetName);
        int numAreaFound = 0;
        int targetArea = 1;
        for(int i=0;i<4;i++){
            if(areaItemCount[i][tarId]>0){
                numAreaFound += 1;
                targetArea = i;
            }
        }
        Log.i("Target", "Target name: " + targetName);
        Log.i("Target", "Target area number found: " + numAreaFound);
        Log.i("Target", "Getting the snapshot of area: " + targetArea);

        try {
            Mat targetSnapshot = approachAndSnapshot(targetArea);
            api.saveMatImage(targetSnapshot, "targetSnapshot.png");
        } catch (IOException e) {
            Log.e("Target", "Error getting the snapshot");
            e.printStackTrace();
        }
        api.takeTargetItemSnapshot();
    }

    @Override
    protected void runPlan2() {
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3() {
        // write your plan 3 here.
    }

    private String yourMethod() {
        return "your method";
    }
}
