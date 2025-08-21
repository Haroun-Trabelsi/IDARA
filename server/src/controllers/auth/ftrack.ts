import { Request, Response, NextFunction } from "express";
import { promises as fs } from "fs";
import path from "path";

export const saveFtrackConfig = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { serverUrl, username, apiKey } = req.body;

    if (!serverUrl || !username || !apiKey) {
      return res.status(400).json({ message: "All fields are required" });
    }

    const envPath = path.resolve(__dirname, "../../../.env");
    const tmpPath = envPath + ".tmp";

    let envContent = "";
    try {
      envContent = await fs.readFile(envPath, "utf8");
    } catch (err: any) {
      if (err.code !== "ENOENT") throw err;
      envContent = "";
    }

    const sanitize = (v: unknown) =>
      String(v).replace(/\r?\n/g, " ").replace(/"/g, '\\"');

    const newValues: Record<string, string> = {
      FTRACK_SERVER_URL: `"${sanitize(serverUrl)}"`,
      FTRACK_USERNAME: `"${sanitize(username)}"`,
      FTRACK_API_KEY: `"${sanitize(apiKey)}"`,
    };

    for (const [key, value] of Object.entries(newValues)) {
      const regex = new RegExp(`(^${key}=).*`, "m");
      if (regex.test(envContent)) {
        envContent = envContent.replace(regex, `${key}=${value}`);
      } else {
        if (envContent.length && !envContent.endsWith("\n")) envContent += "\n";
        envContent += `${key}=${value}\n`;
      }
    }

    await fs.writeFile(tmpPath, envContent.trim() + "\n", "utf8");
    await fs.rename(tmpPath, envPath);

      return res.json({ message: "Ftrack settings saved to .env" });
    } catch (err) {
      next(err);
    }
  };