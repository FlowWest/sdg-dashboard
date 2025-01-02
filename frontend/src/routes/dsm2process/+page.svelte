<script lang="ts">
	import type { PageData } from '../$types.js';
	import type { ActionData } from './$types.js';

	let { data, form }: { data: PageData; form: ActionData } = $props();
	let selected_file = $state<FileList | null>(null);
</script>

<form method="POST" action="?/process_dss" enctype="multipart/form-data">
	<input
		name="dssfile"
		bind:files={selected_file}
		type="file"
		required
		class="file-input file-input-bordered w-full max-w-xs"
	/>

	<button type="submit" class="btn btn-accent">Submit</button>
</form>

<div class="mt-10">
	{#if form && form?.success}
		<p class="text-2xl text-green-500">Successfully uploaded new model!</p>
		<p class="text-2xl text-green-500">{form.message}</p>
	{:else if form && !form.success}
		<p class="text-2xl text-yellow-500">Error trying to submit data</p>
	{/if}
</div>
