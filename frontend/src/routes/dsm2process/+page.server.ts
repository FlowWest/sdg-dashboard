import { env } from '$env/dynamic/private';
import type { Actions } from './$types';

export const actions = {
	process_dss: async ({ request }) => {
		const form_data = await request.formData();
		const file = form_data.get('dssfile');
		console.log({
			typeofFile: typeof file,
			isFile: file instanceof File,
			constructor: file?.constructor?.name,
			fileProps: Object.getOwnPropertyNames(file),
			// If it's a string, what's the content?
			stringContent: typeof file === 'string' ? file : 'not a string'
		});
		if (!file || !(file instanceof File)) {
			return { success: false, error: 'Invalid file upload' };
		}

		const api_form_data = new FormData();
		api_form_data.append('file', file);

		try {
			const resp = await fetch(`${env.API_URL}/uploadfile/`, {
				method: 'POST',
				body: api_form_data
			});
			const resp_data = await resp.json();
			if (resp_data.success) {
				return { success: true, message: resp_data.message };
			} else {
				return { success: false };
			}
		} catch (error) {
			return { error: 'unable to complete request' };
		}
	}
} satisfies Actions;
