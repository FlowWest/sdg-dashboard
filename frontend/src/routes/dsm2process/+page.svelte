<script lang="ts">
	import type { PageData } from '../$types.js';
	import type { ActionData } from './$types.js';

	let { data, form }: { data: PageData; form: ActionData } = $props();
	let selected_file = $state<FileList | null>(null);
</script>

<form method="POST" action="?/process_dss">
	<input
		name="dssfile"
		bind:files={selected_file}
		type="file"
		required
		class="file-input file-input-bordered w-full max-w-xs"
	/>

	<button type="submit" class="btn btn-accent">Submit</button>
</form>

<div class="mt-10 text-2xl text-green-500">
	{#if form?.success}
		<!-- this message is ephemeral; it exists because the page was rendered in
	       response to a form submission. it will vanish if the user reloads -->
		<p>Successfully uploaded new model!</p>
	{/if}
</div>
