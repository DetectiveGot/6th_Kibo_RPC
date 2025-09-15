package jp.jaxa.iss.kibo.rpc.sampleapk;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

public class QuaternionUtils {
//    private float x, y, z, w;
//    QuaternionUtils(float x, float y, float z, float w){
//        this.x = x;
//        this.y = y;
//        this.z = z;
//        this.w = w;
//    }

    public static Quaternion multiply(Quaternion qA, Quaternion qB){
        double aw = qA.getW(), ax = qA.getX(), ay = qA.getY(), az = qA.getZ();
        double bw = qB.getW(), bx = qB.getX(), by = qB.getY(), bz = qB.getZ();

        double rw = aw*bw-ax*bx-ay*by-az*bz;
        double rx = aw*bx+ax*bw-ay*bz+az*by;
        double ry = aw*by+ax*bz+ay*bw-az*bx;
        double rz = aw*bz-ax*by+ay*bx+az*bw;
        return new Quaternion((float)rx, (float)ry, (float)rz, (float)rw);
    }

    public static Quaternion fromAxisAngle(double axisX, double axisY, double axisZ, double angleDeg) {
        double angleRad = Math.toRadians(angleDeg);
        double half = angleRad / 2.0;
        double sin = Math.sin(half);
        double cos = Math.cos(half);

        // Normalize axis
        double mag = Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);
        if (mag == 0) return new Quaternion(0f, 0f, 0f, 1f);  // identity

        double x = axisX / mag * sin;
        double y = axisY / mag * sin;
        double z = axisZ / mag * sin;
        double w = cos;

        return new Quaternion((float)x, (float)y, (float)z, (float)w);
    }


    public static Quaternion fromEulerDegrees(double roll, double pitch, double yaw){ //input degree
        roll = Math.toRadians(roll);
        pitch = Math.toRadians(pitch);
        yaw = Math.toRadians(yaw);
        double cy = Math.cos(yaw/2.0);
        double sy = Math.sin(yaw/2.0);
        double cp = Math.cos(pitch/2.0);
        double sp = Math.sin(pitch/2.0);
        double cr = Math.cos(roll/2.0);
        double sr = Math.sin(roll/2.0);

        double ww = cr*cp*cy+sr*sp*sy;
        double xx = sr*cp*cy-cr*sp*sy;
        double yy = cr*sp*cy+sr*cp*sy;
        double zz = cr*cp*sy-sr*sp*cy;
        return new Quaternion((float)xx, (float)yy, (float)zz, (float)ww);
    }
}
