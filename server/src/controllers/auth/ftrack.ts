import { Request, Response, NextFunction } from "express";
import fs from "fs";
import path from "path";

export const saveFtrackConfig = (req: Request, res: Response, next: NextFunction) => {
  try {
    const { serverUrl, username, apiKey } = req.body;

    if (!serverUrl || !username || !apiKey) {
      return res.status(400).json({ message: "All fields are required" });
    }

    // Path to .env
    const envPath = path.resolve(__dirname, "../../../.env");

    // Read current .env
    let envContent = "";
    if (fs.existsSync(envPath)) {
      envContent = fs.readFileSync(envPath, "utf8");
    }

    // Prepare new key-values
    const newValues: Record<string, string> = {
      FTRACK_SERVER_URL: serverUrl,
      FTRACK_USERNAME: username,
      FTRACK_API_KEY: apiKey,
    };

    // Update or append each variable
    for (const [key, value] of Object.entries(newValues)) {
      const regex = new RegExp(`^${key}=.*$`, "m");
      if (regex.test(envContent)) {
        envContent = envContent.replace(regex, `${key}=${value}`);
      } else {
        envContent += `\n${key}=${value}`;
      }
    }

    // Write back to .env
    fs.writeFileSync(envPath, envContent.trim() + "\n");

    return res.json({ message: "Ftrack settings saved to .env" });
  } catch (err) {
    next(err);
  }
};
