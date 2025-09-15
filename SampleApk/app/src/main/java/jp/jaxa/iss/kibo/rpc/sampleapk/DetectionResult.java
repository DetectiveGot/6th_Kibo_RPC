package jp.jaxa.iss.kibo.rpc.sampleapk;

public class DetectionResult {
    String name;
    float conf;
    DetectionResult(String name, float conf) {
        this.name = name;
        this.conf = conf;
    }
}
