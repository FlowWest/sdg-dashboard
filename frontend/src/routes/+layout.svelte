<script lang="ts">
	import '../app.css';
	let { children } = $props();
</script>

<div class="navbar bg-base-100">
	<div class="navbar-start">
		<div class="dropdown">
			<div tabindex="0" role="button" class="btn btn-ghost lg:hidden">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-5 w-5"
					fill="none"
					viewBox="0 0 24 24"
					stroke="currentColor"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M4 6h16M4 12h8m-8 6h16"
					/>
				</svg>
			</div>
			<ul
				tabindex="0"
				class="menu dropdown-content menu-sm z-[1] mt-3 w-52 rounded-box bg-base-100 p-2 shadow"
			>
				<li><a>Dashboard</a></li>
				<li>
					<a>Tools</a>
					<ul class="p-2">
						<li><a>DSM2 Process</a></li>
						<li><a>Batch 6 to 7</a></li>
					</ul>
				</li>
			</ul>
		</div>
		<a href="/" class="btn btn-ghost text-xl">South Delta Gates</a>
	</div>
	<div class="navbar-center hidden lg:flex">
		<ul class="menu menu-horizontal px-1">
			<li><a href="/">Dashboard</a></li>
			<li>
				<details>
					<summary>Tools</summary>
					<ul class="p-2">
						<li><a href="/dsm2process" class="whitespace-nowrap">DSM2 Process</a></li>
						<li><a class="whitespace-nowrap">Batch 6 to 7</a></li>
					</ul>
				</details>
			</li>
		</ul>
	</div>
</div>

{@render children()}
