package myutil;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Log {
	private String logPath;
	public Log(String logPath) {
		this.logPath = logPath;
	}
	public static void addLogToThisPath(String log, String path) {
		System.out.print(log);
		BufferedWriter bw;
		try {
			bw = new BufferedWriter(new FileWriter(path, true));
			bw.write(log);
			bw.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} // append
	}
	public void addLog(String log) {
		System.out.print(log);
		BufferedWriter bw;
		try {
			bw = new BufferedWriter(new FileWriter(this.logPath, true));
			bw.write(log);
			bw.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} // append
	}
}