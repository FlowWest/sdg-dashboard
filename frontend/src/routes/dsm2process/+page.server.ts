import { env } from '$env/dynamic/private';
import type { Actions } from './$types';

export const actions = {
	process_dss: async ({ request }) => {
		const form_data = await request.formData();
		const file = form_data.get('dssfile') as File;

		if (!file) {
			return { error: 'file processing failed' };
		}

		console.log('file received, sending post request to backend now');
		const api_form_data = new FormData();
		api_form_data.append('file', file);

		try {
			const resp = await fetch(`${env.API_URL}/uploadfile/`, {
				method: 'POST',
				body: api_form_data
			});
			const resp_data = await resp.json();
			if (resp_data.success) {
				return { success: true, message: resp_data };
			} else {
				return { success: false };
			}
		} catch (error) {
			return { error: 'unable to complete request' };
		}
	}
} satisfies Actions;
