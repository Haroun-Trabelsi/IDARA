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

        const assignees = task?.assignments
          ?.map(a => usersById[a.resource?.id])
          .filter(Boolean)
          .join(', ') || 'Unassigned';

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
          assetName: row.asset?.name || ''
        };
      });

    res.json(tasks);
  } catch (err) {
    console.error('❌ Failed to fetch project details:', err);
    res.status(500).json({ error: 'Failed to fetch project details' });
  }
});

router.get('/task/:taskId/components', async (req, res) => {
  const { taskId } = req.params;
  try {
    const sessionInstance = await session;

    // 1. Get the shot ID from the task's parent
    const taskQuery = await sessionInstance.query(`
      select parent.id from Task where id is ${taskId}
    `);

    const shotId = taskQuery.data?.[0]?.parent?.id;
    if (!shotId) {
      return res.status(404).json({ error: 'Shot context not found for task' });
    }

    // 2. Query AssetVersion under that Shot context
    const versions = await sessionInstance.query(`
      select
        asset.name,
        components.name,
        components.file_type,
        components.component_locations.url,
        date
      from AssetVersion
      where asset.parent.id is ${shotId}
      order by date desc
    `);

    const videos = versions.data
      .map((av: any) => ({
        name: av.components?.[0]?.name || av.asset?.name || 'Unnamed',
        fileType: av.components?.[0]?.file_type || '',
        url: av.components?.[0]?.component_locations?.[0]?.url || '',
        date: av.date,
      }))
      .filter((v: any) => v.url);

    res.json(videos);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch video components from shot context' });
  }
});




export default router;
