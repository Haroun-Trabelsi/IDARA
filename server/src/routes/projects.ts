import express from 'express';
import { getFtrackSession } from '../utils/session';
const router = express.Router();
const session = getFtrackSession();

router.get('/projects', async (req, res) => {
    try {
        console.log('Connected to ftrack');
        console.log('About to query projects');
        const response = await (await session).query('select id,name from Project');
        const projects = response.data;

        console.info("Listing " + projects.length + " projects");
        console.log(projects);

        res.json(projects);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to fetch projects' });
    }
});


router.get('/projects/:projectName', async (req, res) => {
  try {
    const { projectName } = req.params;
    console.log(`📦 Fetching asset versions for project ID: ${projectName}`);

    const sessionInstance = await session;

    const query = await sessionInstance.query(`
      select
        asset.name,
        status.name,
        date,
        components.name,
        components.component_locations.url,
        task.id,
        task.name,
        task.type.name,
        task.status.name,
        task.parent.name,
        task.parent.parent.name,
        task.end_date,
        task.bid,
        task.time_logged,
        task.assignments.resource.id
      from AssetVersion
      where task.project.id is ${projectName}
    `);

    const raw = query.data;

    // Extract unique user IDs
    const userIds = [
      ...new Set(
        raw.flatMap(v =>
          v.task?.assignments?.map(a => a.resource?.id).filter(Boolean) || []
        )
      )
    ];

    // Resolve user names
    let usersById: Record<string, string> = {};
    if (userIds.length > 0) {
      const userQuery = await sessionInstance.query(`
        select id, first_name, last_name, username from User where id in (${userIds.join(',')})
      `);
      usersById = Object.fromEntries(
        userQuery.data.map(u => [
          u.id,
          `${u.first_name || ''} ${u.last_name || ''}`.trim() || u.username
        ])
      );
    }

  const versionByTask: Record<string, any[]> = {};
  const seenVersions: Record<string, Set<string>> = {};

  for (const row of raw) {
    const taskId = row.task?.id;
    if (!taskId) continue;

    const status = row.status?.name?.toLowerCase() || '';
    const date = new Date(row.date || 0);
    const url = row.components?.[0]?.component_locations?.[0]?.url;

    if (!url) continue;

    const versionKey = `${status}-${date.toISOString()}`;
    if (!seenVersions[taskId]) seenVersions[taskId] = new Set();
    if (seenVersions[taskId].has(versionKey)) continue;
    seenVersions[taskId].add(versionKey);

    if (!versionByTask[taskId]) versionByTask[taskId] = [];

    versionByTask[taskId].push({
      status,
      date,
      url
    });
  }

    // 🧩 Format into frontend-friendly tasks
    const seenTaskIds = new Set();
const tasks = raw
  .filter(row => {
    const taskId = row.task?.id;
    if (!taskId || seenTaskIds.has(taskId)) return false;
    seenTaskIds.add(taskId);
    return true;
  })
  .map((row, index) => {
    const task = row.task;
    const taskId = task?.id;

    const assignees = task?.assignments
      ?.map(a => usersById[a.resource?.id])
      .filter(Boolean)
      .join(', ') || 'Unassigned';

    // Get all versions for this task
    const versions = versionByTask[taskId] || [];

    // Instead of bestVideo (one), assign all videos:
    const allVideos = versions.map(v => v.url);

    return {
      id: task?.id || '',
      number: index + 1,
      type: `Task (${task?.type?.name || 'Unknown'})`,
      status: task?.status?.name?.toLowerCase() || 'pending',
      assignee: assignees,
      sequence: task?.parent?.parent?.name || 'Unassigned',
      description: task?.parent?.name || 'Unknown',
      dueDate: task?.end_date ? new Date(task.end_date).toISOString().split('T')[0] : null,
      bidHours: task?.bid ? task.bid / 3600 : 0,
      actualHours: task?.time_logged ? task.time_logged / 3600 : 0,
      level: 0,
      icon: task?.type?.name?.toLowerCase() === 'camtrack' ? 'tracking' : undefined,
      assetName: row.asset?.name || '',
      videos: allVideos
    };
  });


    const limitedTasks = tasks.slice(0, 5);
    console.log('✅ Preview of returned tasks:');
    limitedTasks.forEach((task, i) => {
      console.log(`#${i + 1}:`, {
        id: task.id,
        name: task.description,
        status: task.status,
        assignee: task.assignee,
        asset: task.assetName,
        videos: task.videos
      });
    });

    res.json(tasks);
  } catch (err) {
    console.error('❌ Failed to fetch project details with videos:', err);
    res.status(500).json({ error: 'Failed to fetch project details' });
  }
});


export default router;
