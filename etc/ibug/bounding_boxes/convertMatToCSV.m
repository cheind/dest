function convertMatToCSV(file)
	[path,name,ext] = fileparts(file);

	data = load(file);
	m_det = zeros(size(data.bounding_boxes, 2), 4);
	m_gt = zeros(size(data.bounding_boxes, 2), 4);
	for n = 1:size(m_det,1)
		m_det(n,:) = data.bounding_boxes(n){1}.bb_detector;
		m_gt(n,:) = data.bounding_boxes(n){1}.bb_ground_truth;
	end

	f1 = [path name '_det.csv'];
	f2 = [path name '_gt.csv'];
	dlmwrite(f1, m_det, ' ');
	dlmwrite(f2, m_gt, ' ');